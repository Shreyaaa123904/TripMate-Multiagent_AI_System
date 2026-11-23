from datetime import date
from decimal import Decimal
import unittest

from pydantic import ValidationError

from trip_state import (
    LocationGranularity,
    StructuredTripState,
    TravellerCounts,
    TripDates,
    TripLocation,
    TripOperation,
    merge_trip_state,
)


class TripStateModelTests(unittest.TestCase):
    def test_trip_location_country_does_not_require_or_infer_airport(self):
        location = TripLocation(
            raw_text="Japan",
            granularity=LocationGranularity.COUNTRY,
            country="Japan",
        )
        self.assertEqual(location.country, "Japan")
        self.assertIsNone(location.airport_iata)

    def test_invalid_airport_code_rejected(self):
        with self.assertRaises(ValidationError):
            TripLocation(granularity=LocationGranularity.AIRPORT, airport_iata="tokyo")

    def test_new_trip_creates_fresh_context_instead_of_merging_old_trip(self):
        saved = StructuredTripState(
            operation=TripOperation.UPDATE,
            destination=TripLocation(granularity=LocationGranularity.CITY, city="Tokyo"),
            budget_amount=Decimal("1200"),
            budget_currency="USD",
            travellers=TravellerCounts(adults=2),
        )

        incoming = StructuredTripState(
            operation=TripOperation.NEW_TRIP,
            destination=TripLocation(granularity=LocationGranularity.CITY, city="Rome"),
        )

        outcome = merge_trip_state(saved, incoming)

        self.assertTrue(outcome.state_changed)
        self.assertEqual(outcome.merged_state.destination.city, "Rome")
        self.assertIsNone(outcome.merged_state.budget_amount)
        self.assertIsNone(outcome.merged_state.budget_currency)
        self.assertIsNone(outcome.merged_state.travellers)

    def test_partial_date_update_preserves_saved_departure(self):
        saved = StructuredTripState(
            operation=TripOperation.UPDATE,
            dates=TripDates(
                departure_date=date(2026, 12, 1),
                return_date=date(2026, 12, 10),
            ),
        )
        incoming = StructuredTripState(
            operation=TripOperation.UPDATE,
            dates=TripDates(return_date=date(2026, 12, 15)),
        )

        outcome = merge_trip_state(saved, incoming)

        self.assertEqual(outcome.merged_state.dates.departure_date, date(2026, 12, 1))
        self.assertEqual(outcome.merged_state.dates.return_date, date(2026, 12, 15))

    def test_partial_date_update_preserves_saved_return(self):
        saved = StructuredTripState(
            operation=TripOperation.UPDATE,
            dates=TripDates(
                departure_date=date(2026, 12, 1),
                return_date=date(2026, 12, 10),
            ),
        )
        incoming = StructuredTripState(
            operation=TripOperation.UPDATE,
            dates=TripDates(departure_date=date(2026, 12, 3)),
        )

        outcome = merge_trip_state(saved, incoming)

        self.assertEqual(outcome.merged_state.dates.departure_date, date(2026, 12, 3))
        self.assertEqual(outcome.merged_state.dates.return_date, date(2026, 12, 10))

    def test_partial_duration_update_preserves_and_updates_correctly(self):
        saved = StructuredTripState(
            operation=TripOperation.UPDATE,
            dates=TripDates(duration_days=7),
        )
        incoming = StructuredTripState(
            operation=TripOperation.UPDATE,
            dates=TripDates(duration_days=5),
        )

        outcome = merge_trip_state(saved, incoming)

        self.assertEqual(outcome.merged_state.dates.duration_days, 5)

    def test_invalid_partial_date_combination_does_not_overwrite_saved_dates(self):
        saved = StructuredTripState(
            operation=TripOperation.UPDATE,
            dates=TripDates(
                departure_date=date(2026, 12, 10),
                return_date=date(2026, 12, 15),
            ),
        )
        incoming = StructuredTripState(
            operation=TripOperation.UPDATE,
            dates=TripDates(return_date=date(2026, 12, 5)),
        )

        outcome = merge_trip_state(saved, incoming)

        self.assertEqual(outcome.merged_state.dates.departure_date, date(2026, 12, 10))
        self.assertEqual(outcome.merged_state.dates.return_date, date(2026, 12, 15))
        self.assertIn("Dates are invalid after update", " ".join(outcome.clarifications))

    def test_ambiguous_currency_is_preserved_without_default_inference(self):
        saved = StructuredTripState(
            operation=TripOperation.UPDATE,
            budget_amount=Decimal("1500"),
            budget_currency="USD",
        )

        incoming = StructuredTripState(
            operation=TripOperation.UPDATE,
            budget_amount=Decimal("2000"),
            budget_currency_ambiguous=True,
            budget_currency_raw="dollars",
        )

        outcome = merge_trip_state(saved, incoming)

        self.assertEqual(outcome.merged_state.budget_amount, Decimal("2000"))
        self.assertEqual(outcome.merged_state.budget_currency, "USD")
        self.assertTrue(outcome.merged_state.budget_currency_ambiguous)
        self.assertEqual(outcome.merged_state.budget_currency_raw, "dollars")
        self.assertIn("Budget currency is unclear", " ".join(outcome.clarifications))

    def test_definitive_currency_clears_ambiguity_flag_and_raw_value(self):
        saved = StructuredTripState(
            operation=TripOperation.UPDATE,
            budget_currency_ambiguous=True,
            budget_currency_raw="bucks",
        )
        incoming = StructuredTripState(
            operation=TripOperation.UPDATE,
            budget_currency="EUR",
        )

        outcome = merge_trip_state(saved, incoming)

        self.assertEqual(outcome.merged_state.budget_currency, "EUR")
        self.assertFalse(outcome.merged_state.budget_currency_ambiguous)
        self.assertIsNone(outcome.merged_state.budget_currency_raw)

    def test_existing_currency_is_preserved_when_new_currency_is_ambiguous(self):
        saved = StructuredTripState(
            operation=TripOperation.UPDATE,
            budget_currency="USD",
        )
        incoming = StructuredTripState(
            operation=TripOperation.UPDATE,
            budget_currency_ambiguous=True,
            budget_currency_raw="dollar",
        )

        outcome = merge_trip_state(saved, incoming)

        self.assertEqual(outcome.merged_state.budget_currency, "USD")
        self.assertTrue(outcome.merged_state.budget_currency_ambiguous)
        self.assertEqual(outcome.merged_state.budget_currency_raw, "dollar")

    def test_ambiguity_state_persists_when_unrelated_budget_update_arrives(self):
        saved = StructuredTripState(
            operation=TripOperation.UPDATE,
            budget_currency_ambiguous=True,
            budget_currency_raw="dollars",
        )
        incoming = StructuredTripState(
            operation=TripOperation.UPDATE,
            budget_amount=Decimal("2200"),
        )

        outcome = merge_trip_state(saved, incoming)

        self.assertTrue(outcome.merged_state.budget_currency_ambiguous)
        self.assertEqual(outcome.merged_state.budget_currency_raw, "dollars")

    def test_iso_currency_validation_rejects_non_iso_code(self):
        with self.assertRaises(ValidationError):
            StructuredTripState(operation=TripOperation.UPDATE, budget_currency="ZZZ")

    def test_dates_validated_before_merging_new_values(self):
        with self.assertRaises(ValidationError):
            TripDates(
                departure_date=date(2026, 12, 10),
                return_date=date(2026, 12, 8),
            )

    def test_valid_saved_dates_remain_when_new_dates_not_supplied(self):
        saved = StructuredTripState(
            operation=TripOperation.UPDATE,
            dates=TripDates(
                departure_date=date(2026, 12, 1),
                return_date=date(2026, 12, 10),
            ),
        )

        incoming = StructuredTripState(operation=TripOperation.UPDATE)
        outcome = merge_trip_state(saved, incoming)

        self.assertIsNotNone(outcome.merged_state.dates)
        self.assertEqual(outcome.merged_state.dates.departure_date, date(2026, 12, 1))
        self.assertEqual(outcome.merged_state.dates.return_date, date(2026, 12, 10))

    def test_ambiguous_traveller_count_does_not_overwrite_saved_counts(self):
        saved = StructuredTripState(
            operation=TripOperation.UPDATE,
            travellers=TravellerCounts(adults=2, children=1),
        )

        incoming = StructuredTripState(
            operation=TripOperation.UPDATE,
            travellers=TravellerCounts(ambiguous_text="family"),
        )

        outcome = merge_trip_state(saved, incoming)

        self.assertIsNotNone(outcome.merged_state.travellers)
        self.assertEqual(outcome.merged_state.travellers.adults, 2)
        self.assertEqual(outcome.merged_state.travellers.children, 1)
        self.assertIn("Traveller count is ambiguous", " ".join(outcome.clarifications))

    def test_rooms_must_be_positive_when_supplied(self):
        with self.assertRaises(ValidationError):
            TravellerCounts(rooms=0)

        with self.assertRaises(ValidationError):
            TravellerCounts(rooms=-1)

    def test_partial_traveller_update_preserves_saved_adults(self):
        saved = StructuredTripState(
            operation=TripOperation.UPDATE,
            travellers=TravellerCounts(adults=2, children=1, infants=0, rooms=1),
        )
        incoming = StructuredTripState(
            operation=TripOperation.UPDATE,
            travellers=TravellerCounts(children=2),
        )

        outcome = merge_trip_state(saved, incoming)

        self.assertEqual(outcome.merged_state.travellers.adults, 2)
        self.assertEqual(outcome.merged_state.travellers.children, 2)
        self.assertEqual(outcome.merged_state.travellers.infants, 0)
        self.assertEqual(outcome.merged_state.travellers.rooms, 1)

    def test_partial_traveller_update_preserves_saved_children_when_rooms_change(self):
        saved = StructuredTripState(
            operation=TripOperation.UPDATE,
            travellers=TravellerCounts(adults=2, children=1, infants=1, rooms=1),
        )
        incoming = StructuredTripState(
            operation=TripOperation.UPDATE,
            travellers=TravellerCounts(rooms=2),
        )

        outcome = merge_trip_state(saved, incoming)

        self.assertEqual(outcome.merged_state.travellers.adults, 2)
        self.assertEqual(outcome.merged_state.travellers.children, 1)
        self.assertEqual(outcome.merged_state.travellers.infants, 1)
        self.assertEqual(outcome.merged_state.travellers.rooms, 2)

    def test_invalid_partial_traveller_combination_does_not_overwrite_saved_counts(self):
        saved = StructuredTripState(
            operation=TripOperation.UPDATE,
            travellers=TravellerCounts(adults=1, children=1),
        )
        incoming = StructuredTripState(
            operation=TripOperation.UPDATE,
            travellers=TravellerCounts(adults=0),
        )

        outcome = merge_trip_state(saved, incoming)

        self.assertEqual(outcome.merged_state.travellers.adults, 1)
        self.assertEqual(outcome.merged_state.travellers.children, 1)
        self.assertIn("Traveller counts are invalid after update", " ".join(outcome.clarifications))

    def test_unsupported_remove_operation_rejected(self):
        with self.assertRaises(ValidationError):
            StructuredTripState(operation=TripOperation.REMOVE, remove_fields=["hotels"])

    def test_supported_remove_operation_clears_only_requested_field(self):
        saved = StructuredTripState(
            operation=TripOperation.UPDATE,
            budget_amount=Decimal("1000"),
            budget_currency="EUR",
            travellers=TravellerCounts(adults=2),
        )

        incoming = StructuredTripState(
            operation=TripOperation.REMOVE,
            remove_fields=["budget"],
        )

        outcome = merge_trip_state(saved, incoming)

        self.assertIsNone(outcome.merged_state.budget_amount)
        self.assertIsNone(outcome.merged_state.budget_currency)
        self.assertIsNotNone(outcome.merged_state.travellers)
        self.assertEqual(outcome.merged_state.travellers.adults, 2)

    def test_reset_requires_explicit_supported_scope(self):
        with self.assertRaises(ValidationError):
            StructuredTripState(operation=TripOperation.RESET, reset_scope="unsupported")

    def test_remove_operation_requires_fields(self):
        with self.assertRaises(ValidationError):
            StructuredTripState(operation=TripOperation.REMOVE)


if __name__ == "__main__":
    unittest.main()
