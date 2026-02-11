from decimal import Decimal, ROUND_HALF_EVEN, InvalidOperation
from typing import Union, Optional

Number = Union[int, float, str, Decimal]


class Calculator:
    """
    Stateful calculator with a small public API:
      - add(value)
      - subtract(value)
      - multiply(value)
      - divide(value)
      - percent(value)
      - percent_add(value)
      - percent_subtract(value)
      - get_total()
      - clear()       # one-shot undo (restore previous total, then forget it)
      - clear_all()   # wipe everything

    Internals (_total, _last_total) are intentionally non-public.
    """

    def __init__(self, precision: int = 2, max_value: Number = 1000):
        self.precision = int(precision)
        self.max_value = self._to_decimal(max_value)
        # store totals as already-quantized Decimals
        zero = Decimal("0").quantize(Decimal("1").scaleb(-self.precision))
        self._total: Decimal = zero
        self._last_total: Optional[Decimal] = None  # None == no undo available

    # Public API ---------------------------------------------------------

    def get_total(self) -> Decimal:
        """Return current total as Decimal (quantized to precision)."""
        return self._total

    def clear(self) -> Decimal:
        """
        One-shot undo: restore previous total if available, then forget the last_total.
        If no undo is available this is a no-op.
        Returns the new current total.
        """
        if self._last_total is None:
            # no undo available; do nothing
            return self._total

        # restore last_total and remove undo (single-use)
        restored = self._last_total
        self._last_total = None
        self._total = restored
        return self._total

    def clear_all(self) -> Decimal:
        """
        Clear everything: reset both current total and last total to zero and remove undo.
        Returns the new current total.
        """
        zero = Decimal("0").quantize(Decimal("1").scaleb(-self.precision))
        self._last_total = None
        self._total = zero
        return self._total

    def add(self, value: Number) -> Decimal:
        return self._apply_delta(self._to_decimal(value))

    def subtract(self, value: Number) -> Decimal:
        return self._apply_delta(-self._to_decimal(value))

    def multiply(self, factor: Number) -> Decimal:
        dec_factor = self._to_decimal(factor)
        new_total = self._quantize(self._total * dec_factor)
        return self._set_total_with_check(new_total)

    def divide(self, divisor: Number) -> Decimal:
        dec_div = self._to_decimal(divisor)
        if dec_div == 0:
            raise ValueError("Division by zero")
        new_total = self._quantize(self._total / dec_div)
        return self._set_total_with_check(new_total)

    def percent(self, value: Number) -> Decimal:
        """Set total to (total * value / 100) and return new total.

        Accepts the same input types as other operations; result is quantized and
        checked against max_value. Undo (clear) will restore the previous total.
        """
        dec_value = self._to_decimal(value)
        new_total = self._quantize(self._total * dec_value / Decimal("100"))
        return self._set_total_with_check(new_total)

    def percent_add(self, value: Number) -> Decimal:
        """Add (total * value / 100) to the current total and return new total.

        Accepts the same input types as other operations; result is quantized and
        checked against max_value. Undo (clear) will restore the previous total.
        """
        dec_value = self._to_decimal(value)
        increment = (self._total * dec_value) / Decimal("100")
        new_total = self._quantize(self._total + increment)
        return self._set_total_with_check(new_total)

    def percent_substract(self, value: Number) -> Decimal:
        """Subtract (total * value / 100) from the current total and return new total.

        Accepts the same input types as other operations; result is quantized and
        checked against max_value. Undo (clear) will restore the previous total.
        """
        dec_value = self._to_decimal(value)
        decrement = (self._total * dec_value) / Decimal("100")
        new_total = self._quantize(self._total - decrement)
        return self._set_total_with_check(new_total)

    # Internal helpers (non-public) -------------------------------------

    def _apply_delta(self, delta: Decimal) -> Decimal:
        new_total = self._quantize(self._total + delta)
        return self._set_total_with_check(new_total)

    def _set_total_with_check(self, new_total: Decimal) -> Decimal:
        """
        Atomically check max_value, update last_total then commit or restore.
        Raises ValueError on overflow.
        """
        prev = self._total
        # record previous total so it can be restored by a single clear()
        self._last_total = prev
        # treat max_value as a magnitude limit
        if abs(new_total) > self.max_value:
            self._total = prev
            raise ValueError("Total max value reached")
        self._total = new_total
        return self._total

    def _quantize(self, value: Decimal) -> Decimal:
        """Round value to configured precision using ROUND_HALF_EVEN."""
        exp = Decimal("1").scaleb(-self.precision)  # e.g. precision=2 -> Decimal('0.01')
        return value.quantize(exp, rounding=ROUND_HALF_EVEN)

    def _to_decimal(self, value: Number) -> Decimal:
        """Convert accepted inputs to Decimal.

        Accepts: Decimal, int, float (converted via str), and decimal-like str.
        Rejects: bytes, None, arbitrary objects, and non-decimal numeric strings
                 (e.g. '0x10' will raise ValueError).
        """
        if isinstance(value, Decimal):
            return value
        if isinstance(value, int):
            return Decimal(value)
        if isinstance(value, float):
            # convert via str to avoid capturing binary float artifacts
            return Decimal(str(value))
        if isinstance(value, str):
            try:
                return Decimal(value)
            except InvalidOperation as exc:
                raise ValueError(f"Invalid numeric string: {value!r}") from exc
        raise TypeError(f"Unsupported value type: {type(value).__name__}")
