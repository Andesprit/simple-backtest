"""Portfolio management for positions, cash, and trade history."""

from copy import deepcopy
from datetime import datetime
from math import isfinite
from typing import Any, Dict, List, Optional
from uuid import uuid4


class Portfolio:
    """Tracks cash, positions, and trade history with FIFO position management.

    This portfolio models one long-only, cash-funded instrument. ``shares`` is
    a generic fractional quantity, but margin, shorting, leverage, contract
    multipliers, financing, and currency conversion are not modeled.
    """

    def __init__(self, initial_capital: float):
        """Initialize portfolio.

        :param initial_capital: Starting cash (must be positive)
        """
        if not isfinite(initial_capital) or initial_capital <= 0:
            raise ValueError(f"initial_capital must be positive, got {initial_capital}")

        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.trade_history: List[Dict[str, Any]] = []
        self._total_shares = 0.0

    def get_total_shares(self) -> float:
        """Return total units/quantity held across all positions.

        Note: 'shares' here means generic units (stocks, lots, coins, contracts, etc.)
        """
        return self._total_shares

    def get_portfolio_value(self, current_price: float) -> float:
        """Calculate total portfolio value.

        :param current_price: Current market price per unit
        :return: Cash + position values
        """
        if not isfinite(current_price) or current_price < 0:
            raise ValueError(f"current_price must be finite and non-negative, got {current_price}")
        position_value = self.get_total_shares() * current_price
        return self.cash + position_value

    def can_afford(self, shares: float, price: float, commission: float) -> bool:
        """Check if sufficient cash for purchase.

        :param shares: Units/quantity to buy (fractional allowed)
        :param price: Price per unit
        :param commission: Commission cost
        :return: True if affordable
        """
        total_cost = (shares * price) + commission
        return self.cash >= total_cost

    def execute_buy(
        self,
        shares: float,
        price: float,
        commission: float,
        timestamp: datetime,
        order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute buy order and update portfolio.

        :param shares: Units/quantity to buy (positive, fractional allowed)
        :param price: Price per unit (positive)
        :param commission: Commission cost (non-negative)
        :param timestamp: Execution time
        :param order_id: Order ID (auto-generated if None)
        :return: Trade info dict
        """
        self._validate_order_values(shares, price, commission)

        total_cost = (shares * price) + commission

        if not self.can_afford(shares, price, commission):
            raise ValueError(f"Insufficient cash. Need {total_cost:.2f}, have {self.cash:.2f}")

        # Generate order ID if not provided
        if order_id is None:
            order_id = f"BUY_{uuid4().hex[:8]}"
        elif order_id in self.positions:
            raise ValueError(f"Position with order_id '{order_id}' already exists")

        # Deduct cash
        self.cash -= total_cost

        # Add position
        self.positions[order_id] = {
            "shares": shares,
            "entry_price": price,
            "entry_time": timestamp,
            "entry_commission": commission,
        }
        self._total_shares += shares

        # Create trade record
        trade_info = {
            "order_id": order_id,
            "timestamp": timestamp,
            "signal": "buy",
            "shares": shares,
            "price": price,
            "commission": commission,
            "portfolio_value": self.get_portfolio_value(price),
            "cash": self.cash,
            "positions": deepcopy(self.positions),
            "pnl": None,
        }

        self.trade_history.append(trade_info)
        return trade_info

    def execute_sell(
        self,
        shares: float,
        price: float,
        commission: float,
        timestamp: datetime,
        order_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Execute sell order using FIFO or specified order IDs.

        :param shares: Units/quantity to sell (positive, fractional allowed)
        :param price: Price per unit (positive)
        :param commission: Commission cost (non-negative)
        :param timestamp: Execution time
        :param order_ids: Specific orders to sell from (FIFO if None)
        :return: Trade info dict with P&L
        """
        self._validate_order_values(shares, price, commission)

        total_shares_held = self.get_total_shares()
        if shares > total_shares_held:
            raise ValueError(
                f"Insufficient shares. Trying to sell {shares}, have {total_shares_held}"
            )

        if self.cash + shares * price < commission:
            raise ValueError("Insufficient cash and proceeds to pay sell commission")

        # Determine which positions to sell from
        if order_ids is not None:
            if len(order_ids) != len(set(order_ids)):
                raise ValueError("order_ids contains duplicate order_ids")
            # Validate specified order_ids exist
            invalid_ids = set(order_ids) - set(self.positions.keys())
            if invalid_ids:
                raise ValueError(f"Invalid order_ids specified: {invalid_ids}")
            sell_order_ids = order_ids
            selected_shares = sum(self.positions[order_id]["shares"] for order_id in order_ids)
            if selected_shares < shares:
                raise ValueError(
                    f"Selected order_ids contain only {selected_shares:g} shares, "
                    f"cannot sell {shares:g}"
                )
        else:
            # Use FIFO: sort by entry_time
            sell_order_ids = sorted(
                self.positions.keys(), key=lambda oid: self.positions[oid]["entry_time"]
            )

        # Sell shares from positions
        shares_remaining = shares
        total_pnl = 0.0
        closed_order_ids = []
        fills = []

        for order_id in sell_order_ids:
            if shares_remaining <= 0:
                break

            position = self.positions[order_id]
            position_shares = position["shares"]
            entry_price = position["entry_price"]
            entry_commission = position.get("entry_commission", 0.0)

            # Calculate how much to sell from this position
            shares_to_sell = min(shares_remaining, position_shares)

            # Calculate P&L for this portion
            entry_commission_allocated = entry_commission * (shares_to_sell / position_shares)
            pnl = shares_to_sell * (price - entry_price) - entry_commission_allocated
            total_pnl += pnl

            # Update position
            position["shares"] -= shares_to_sell
            position["entry_commission"] -= entry_commission_allocated
            shares_remaining -= shares_to_sell
            fills.append(
                {
                    "order_id": order_id,
                    "shares": shares_to_sell,
                    "entry_price": entry_price,
                    "entry_commission": entry_commission_allocated,
                    "pnl_before_exit_commission": pnl,
                }
            )

            # Remove position if fully closed
            if position["shares"] == 0:
                closed_order_ids.append(order_id)

        # Remove closed positions
        for order_id in closed_order_ids:
            del self.positions[order_id]

        self._total_shares -= shares

        # Add proceeds to cash (minus commission)
        proceeds = (shares * price) - commission
        self.cash += proceeds

        # Adjust total P&L for commission
        total_pnl -= commission

        # Create trade record
        trade_info = {
            "order_id": f"SELL_{uuid4().hex[:8]}",
            "timestamp": timestamp,
            "signal": "sell",
            "shares": shares,
            "price": price,
            "commission": commission,
            "portfolio_value": self.get_portfolio_value(price),
            "cash": self.cash,
            "positions": deepcopy(self.positions),
            "pnl": total_pnl,
            "fills": fills,
        }

        self.trade_history.append(trade_info)
        return trade_info

    def get_trade_history(self) -> List[Dict[str, Any]]:
        """Return copy of trade history."""
        return deepcopy(self.trade_history)

    def reset(self) -> None:
        """Reset to initial state."""
        self.cash = self.initial_capital
        self.positions.clear()
        self.trade_history.clear()
        self._total_shares = 0.0

    def get_state_snapshot(self) -> Dict[str, Any]:
        """Return current state snapshot."""
        return {
            "cash": self.cash,
            "positions": deepcopy(self.positions),
            "total_shares": self.get_total_shares(),
        }

    @staticmethod
    def _validate_order_values(shares: float, price: float, commission: float) -> None:
        """Validate numeric order inputs before any accounting mutation."""
        values = {"shares": shares, "price": price, "commission": commission}
        for name, value in values.items():
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number, got {value}")
        if shares <= 0:
            raise ValueError(f"shares must be positive, got {shares}")
        if price <= 0:
            raise ValueError(f"price must be positive, got {price}")
        if commission < 0:
            raise ValueError(f"commission must be non-negative, got {commission}")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Portfolio(cash={self.cash:.2f}, "
            f"positions={len(self.positions)}, "
            f"total_shares={self.get_total_shares():.2f})"
        )
