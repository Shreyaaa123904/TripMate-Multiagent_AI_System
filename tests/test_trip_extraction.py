import unittest

from trip_extraction import extract_structured_trip
from trip_state import TripOperation


class _FakeResponse:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    def __init__(self, content: str):
        self._content = content
        self.calls = 0
        self.last_messages = None

    def invoke(self, messages):
        self.calls += 1
        self.last_messages = messages
        return _FakeResponse(self._content)


class TripExtractionTests(unittest.TestCase):
    def test_new_trip_extraction_success(self):
        llm = _FakeLLM(
            '{"operation":"new_trip","destination":{"raw_text":"Paris","granularity":"city","city":"Paris"},"origin":{"raw_text":"New York","granularity":"city","city":"New York"},"dates":{"departure_date":"2027-03-12","return_date":"2027-03-20","duration_days":8},"travellers":{"adults":2,"rooms":1},"budget_amount":2500,"budget_currency":"USD","preferences":["museum"],"clarifications":[]}'
        )

        result = extract_structured_trip("Plan a new trip from New York to Paris", llm=llm)

        self.assertIsNotNone(result.state)
        self.assertEqual(result.state.operation, TripOperation.NEW_TRIP)
        self.assertEqual(result.state.destination.city, "Paris")
        self.assertEqual(result.state.travellers.rooms, 1)
        self.assertEqual(result.state.dates.duration_days, 8)
        self.assertEqual(llm.calls, 1)

    def test_update_trip_partial_extraction(self):
        llm = _FakeLLM(
            '{"operation":"update_trip","travellers":{"children":1},"clarifications":[]}'
        )

        result = extract_structured_trip("Add one child to the trip", llm=llm)

        self.assertIsNotNone(result.state)
        self.assertEqual(result.state.operation, TripOperation.UPDATE)
        self.assertEqual(result.state.travellers.children, 1)
        self.assertIsNone(result.state.travellers.adults)

    def test_follow_up_question_intent(self):
        llm = _FakeLLM('{"operation":"follow_up_question","clarifications":[]}')

        result = extract_structured_trip("Is this plan okay for rainy season?", llm=llm)

        self.assertIsNotNone(result.state)
        self.assertEqual(result.state.operation, TripOperation.FOLLOW_UP)

    def test_remove_fields_intent(self):
        llm = _FakeLLM('{"operation":"remove_fields","remove_fields":["budget","dates"],"clarifications":[]}')

        result = extract_structured_trip("Remove budget and dates", llm=llm)

        self.assertIsNotNone(result.state)
        self.assertEqual(result.state.operation, TripOperation.REMOVE)
        self.assertEqual(result.state.remove_fields, ["budget", "dates"])

    def test_reset_intent(self):
        llm = _FakeLLM('{"operation":"reset","reset_scope":"trip","clarifications":[]}')

        result = extract_structured_trip("Start over", llm=llm)

        self.assertIsNotNone(result.state)
        self.assertEqual(result.state.operation, TripOperation.RESET)
        self.assertEqual(result.state.reset_scope, "trip")

    def test_ambiguous_values_require_clarification(self):
        llm = _FakeLLM(
            '{"operation":"auto","destination":{"raw_text":"Japan","granularity":"unknown"},"budget_amount":2000,"budget_currency_ambiguous":true,"budget_currency_raw":"dollars","clarifications":[]}'
        )

        result = extract_structured_trip("Plan Japan trip under 2000 dollars", llm=llm)

        self.assertIsNotNone(result.state)
        joined = " ".join(result.clarifications)
        self.assertIn("Destination is unclear", joined)
        self.assertIn("Budget currency is unclear", joined)
        self.assertIsNone(result.state.destination.country)
        self.assertIsNone(result.state.destination.city)

    def test_unknown_tokens_are_normalized_not_invented(self):
        llm = _FakeLLM(
            '{"operation":"auto","origin":{"raw_text":"unknown","granularity":"unknown"},"destination":{"raw_text":"unknown","granularity":"unknown"},"clarifications":[]}'
        )

        result = extract_structured_trip("some unclear request", llm=llm)

        self.assertIsNotNone(result.state)
        self.assertIsNone(result.state.origin.raw_text)
        self.assertIsNone(result.state.destination.raw_text)

    def test_invalid_currency_fails_validation(self):
        llm = _FakeLLM('{"operation":"update_trip","budget_currency":"ZZZ","clarifications":[]}')

        result = extract_structured_trip("Set currency to ZZZ", llm=llm)

        self.assertIsNone(result.state)
        self.assertEqual(result.extraction_error, "validation_error")
        self.assertTrue(any("budget_currency" in item for item in result.clarifications))

    def test_malformed_llm_json_returns_clarification(self):
        llm = _FakeLLM("This is not valid JSON")

        result = extract_structured_trip("Trip to somewhere", llm=llm)

        self.assertIsNone(result.state)
        self.assertTrue(result.extraction_error and result.extraction_error.startswith("json_parse_error:"))
        self.assertGreater(len(result.clarifications), 0)

    def test_empty_message(self):
        llm = _FakeLLM('{"operation":"auto"}')

        result = extract_structured_trip("   ", llm=llm)

        self.assertIsNone(result.state)
        self.assertEqual(result.extraction_error, "empty_message")
        self.assertEqual(llm.calls, 0)


if __name__ == "__main__":
    unittest.main()
