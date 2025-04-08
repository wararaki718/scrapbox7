from unittest import TestCase

from app.sample import (
    sample_output_info_log,
    sample_raise_exception,
    sample_raise_exception_and_error_log,
)


class LogTest(TestCase):
    def test_output_info_log(self) -> None:
        with self.assertLogs("app.sample", level="INFO") as cm:
            sample_output_info_log()
        self.assertEqual(cm.output, ["INFO:app.sample:this is info"])


    def test_output_exception_log(self) -> None:
        with self.assertRaises(Exception) as cm:
            sample_raise_exception()
            self.assertIsInstance(cm.exception, Exception)
            self.assertEqual(str(cm.exception), "this is an exception")

    def test_output_exception_and_error_log(self) -> None:
        with self.assertLogs("app.sample", level="ERROR") as log:
            with self.assertRaises(Exception) as cm:
                sample_raise_exception_and_error_log()

                self.assertIsInstance(cm.exception, Exception)
                self.assertEqual(str(cm.exception), "this is an exception")

            self.assertEqual(log.output, ["ERROR:app.sample:this is an error"])
