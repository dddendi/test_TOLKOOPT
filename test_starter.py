"""Проверки правил задания. Запуск: python -m unittest -v"""

import asyncio
import unittest
from decimal import Decimal
from unittest.mock import AsyncMock, patch

from starter import recommend_price, sample_fetch_offers


def offer(price="4800.00", days=3, seller="PARTS"):
    return {"seller": seller, "price": Decimal(price), "delivery_days": days}


class RecommendPriceTests(unittest.IsolatedAsyncioTestCase):
    async def calculate(self, offers, current="4900.00", minimum="4500.00", status="success"):
        fetch = AsyncMock(return_value={"status": status, "offers": offers})
        result = await recommend_price(fetch, Decimal(current), Decimal(minimum))
        fetch.assert_awaited_once_with()
        self.assertEqual(set(result), {"price", "decision", "reason"})
        self.assertIsInstance(result["price"], Decimal)
        self.assertIsInstance(result["reason"], str)
        self.assertTrue(result["reason"].strip())
        return result

    def assert_hold(self, result, current="4900.00"):
        self.assertEqual(result["price"], Decimal(current))
        self.assertEqual(result["decision"], "HOLD")

    async def test_original_example(self):
        """Исходный пример: исключить свой магазин и долгую доставку, получить 4728.00."""
        result = await self.calculate([
            offer("4300.00", 2, "myshop"),
            offer("4800.00", 3),
            offer("4700.00", 8, "SLOW"),
        ])
        self.assertEqual(result["price"], Decimal("4728.00"))
        self.assertEqual(result["decision"], "APPLY")

    async def test_own_shop_is_excluded_in_any_case(self):
        """MYSHOP, myshop и смешанный регистр не должны влиять на минимум."""
        for seller in ("MYSHOP", "myshop", "MyShop", "mYsHoP"):
            with self.subTest(seller=seller):
                result = await self.calculate([offer("100.00", seller=seller), offer()])
                self.assertEqual(result["price"], Decimal("4728.00"))

    async def test_delivery_boundaries_and_unknown_delivery(self):
        """Доставка 0 и 6 дней подходит; 7 дней и None исключаются."""
        for days in (0, 6):
            with self.subTest(days=days):
                result = await self.calculate([
                    offer("100.00", None), offer("200.00", 7), offer(days=days),
                ])
                self.assertEqual(result["price"], Decimal("4728.00"))

    async def test_no_suitable_offers_keeps_price_even_below_minimum(self):
        """Пустой ответ или все предложения отфильтрованы: HOLD без повышения до min_price."""
        cases = {
            "empty": [],
            "slow": [offer(days=7)],
            "unknown": [offer(days=None)],
            "own_shop": [offer(seller="MYSHOP")],
        }
        for name, offers in cases.items():
            with self.subTest(case=name):
                result = await self.calculate(offers, current="4000.00")
                self.assert_hold(result, "4000.00")

    async def test_unsuccessful_status_keeps_current_price(self):
        """partial и error приводят к HOLD, даже если есть предложения и цена ниже min_price."""
        for status in ("partial", "error"):
            with self.subTest(status=status):
                result = await self.calculate([offer()], current="4000.00", status=status)
                self.assert_hold(result, "4000.00")

    async def test_source_exceptions_keep_current_price(self):
        """Исключения источника, включая TypeError, превращаются в HOLD."""
        for error in (RuntimeError("сбой"), ConnectionError("нет связи"), TypeError("сбой источника")):
            with self.subTest(error=type(error).__name__):
                fetch = AsyncMock(side_effect=error)
                result = await recommend_price(fetch, Decimal("4000.00"), Decimal("4500.00"))
                self.assert_hold(result, "4000.00")

    async def test_timeout_cancels_waiting_source(self):
        """Источник, который не завершился вовремя, отменяется; возвращается HOLD."""
        cancelled = asyncio.Event()

        async def waiting_source():
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        result = await recommend_price(
            waiting_source, Decimal("4000.00"), Decimal("4500.00"), timeout_s=0.01,
        )
        self.assert_hold(result, "4000.00")
        self.assertTrue(cancelled.is_set())

    async def test_decimal_rounds_down_to_kopeck(self):
        """100.01 × 0.985 = 98.50985: нужно 98.50, а не округление до 98.51."""
        result = await self.calculate([offer("100.01")], current="100.00", minimum="1.00")
        self.assertEqual(result["price"], Decimal("98.50"))
        self.assertEqual(result["price"].as_tuple().exponent, -2)

    async def test_minimum_price_is_respected(self):
        """Если цена после скидки ниже min_price, рекомендация равна min_price."""
        result = await self.calculate([offer("4000.00")])
        self.assertEqual(result["price"], Decimal("4500.00"))
        self.assertEqual(result["decision"], "APPLY")

    async def test_change_threshold_in_both_directions(self):
        """От текущих 985: ровно ±30% допустимо, выход за границу требует REVIEW."""
        cases = [
            ("699.99", "689.49", "REVIEW"),
            ("700.00", "689.50", "APPLY"),
            ("700.01", "689.50", "APPLY"),
            ("1000.00", "985.00", "APPLY"),
            ("1299.99", "1280.49", "APPLY"),
            ("1300.00", "1280.50", "APPLY"),
            ("1300.01", "1280.50", "APPLY"),
            ("1300.02", "1280.51", "REVIEW"),
        ]
        for competitor_price, expected_price, decision in cases:
            with self.subTest(competitor_price=competitor_price):
                result = await self.calculate(
                    [offer(competitor_price)], current="985.00", minimum="1.00",
                )
                self.assertEqual(result["price"], Decimal(expected_price))
                self.assertEqual(result["decision"], decision)

    async def test_review_uses_price_after_minimum_is_applied(self):
        """Порог 30% проверяется после применения min_price; REVIEW сохраняет рекомендацию."""
        result = await self.calculate([offer("100.00")], current="100.00", minimum="150.00")
        self.assertEqual(result["price"], Decimal("150.00"))
        self.assertEqual(result["decision"], "REVIEW")

    async def test_caller_cancellation_is_not_hidden(self):
        """Отмена задачи извне должна распространяться, а не превращаться в HOLD."""
        fetch = AsyncMock(side_effect=asyncio.CancelledError())
        with self.assertRaises(asyncio.CancelledError):
            await recommend_price(fetch, Decimal("4900.00"), Decimal("4500.00"))

    async def test_sample_random_delay_is_awaited(self):
        """Пример выбирает задержку от 0 до 3 секунд и ожидает её через asyncio.sleep."""
        with patch("starter.random.uniform", return_value=1.5) as uniform:
            with patch("starter.asyncio.sleep", new_callable=AsyncMock) as sleep:
                result = await sample_fetch_offers()
        uniform.assert_called_once_with(0, 3)
        sleep.assert_awaited_once_with(1.5)
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["offers"]), 3)

    async def test_sample_fast_response_and_timeout(self):
        """Тот же пример при задержке 0 даёт APPLY, при задержке 3 — HOLD по тайм-ауту."""
        for delay, decision in ((0, "APPLY"), (3, "HOLD")):
            with self.subTest(delay=delay):
                with patch("starter.random.uniform", return_value=delay):
                    result = await recommend_price(
                        sample_fetch_offers, Decimal("4900.00"), Decimal("4500.00"), timeout_s=0.05,
                    )
                self.assertEqual(result["decision"], decision)
                expected = "4728.00" if decision == "APPLY" else "4900.00"
                self.assertEqual(result["price"], Decimal(expected))


if __name__ == "__main__":
    unittest.main()
