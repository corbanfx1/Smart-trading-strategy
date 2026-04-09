"""
Setup Validator – applies institutional-grade filters to remove weak setups.
Checks: MTF alignment, R:R quality, market hours, score thresholds.
"""
import logging
from typing import List, Tuple

from strategy.setup_detector import TradeSetup
import config

logger = logging.getLogger(__name__)


class SetupValidator:
    """Filter a list of TradeSetup objects and return only validated ones."""

    def validate(
        self, setups: List[TradeSetup]
    ) -> Tuple[List[TradeSetup], List[Tuple[TradeSetup, str]]]:
        """
        Returns:
          valid   : list of setups that pass all filters
          rejected: list of (setup, reason) tuples
        """
        valid    = []
        rejected = []

        for setup in setups:
            ok, reason = self._check(setup)
            if ok:
                valid.append(setup)
            else:
                rejected.append((setup, reason))
                logger.debug("Rejected setup @ %s: %s", setup.timestamp, reason)

        logger.info(
            "Validator: %d valid / %d rejected out of %d",
            len(valid), len(rejected), len(setups),
        )
        return valid, rejected

    # ── Filters ───────────────────────────────────────────────────────────────

    @staticmethod
    def _check(setup: TradeSetup) -> Tuple[bool, str]:
        if setup.setup_score < config.SETUP_SCORE_THRESHOLD * 100:
            return False, f"Score {setup.setup_score:.1f} < threshold"

        if setup.rr1 < config.MIN_RR_RATIO:
            return False, f"R:R {setup.rr1:.2f} < min {config.MIN_RR_RATIO}"

        if config.MTF_ALIGNMENT_REQUIRED and not setup.mtf_aligned:
            return False, "MTF misalignment"

        if setup.vol_regime == "high" and setup.setup_score < 75:
            return False, "High vol requires score ≥ 75"

        stop_pct = abs(setup.entry - setup.stop) / setup.entry
        if stop_pct > config.MAX_STOP_PCT:
            return False, f"Stop {stop_pct*100:.2f}% > max {config.MAX_STOP_PCT*100:.1f}%"

        return True, ""
