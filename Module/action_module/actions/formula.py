from openpyxl import load_workbook, worksheet, Workbook
from openpyxl.worksheet.datavalidation import Unique
from openpyxl.drawing.image import Image
import openpyxl as excel
from helper_functions import excel_column_to_index
import pandas as pd
from openpyxl.workbook.defined_name import DefinedName
from openpyxl.utils import get_column_letter, column_index_from_string
from openpyxl.utils.cell import coordinate_from_string
from asposecells.api import ProtectionType as pt, Workbook as wbs
import datetime

class formula:
    def __init__(self):
        return
    def sum(workbook_path, sheet_name):
        pass
    import datetime

    def today():
        return datetime.date.today()

    def now():
        return datetime.datetime.now()

    def create_date(year, month, day):
        try:
            return datetime.date(year, month, day)
        except ValueError:
            print("Invalid date provided. Please enter a valid year, month, and day.")
            return None

    def create_time(hour, minute, second):
        """Creates a time value given an hour, minute, and second."""
        try:
            return datetime.time(hour, minute, second)
        except ValueError:
            print("Invalid time provided. Please enter valid hour, minute, and second values (0-23, 0-59, 0-59).")
            return None

    def extract_year(date_obj):
        """Extracts the year from a given date object."""
        if isinstance(date_obj, datetime.date):
            return date_obj.year
        else:
            print("Invalid input. Please provide a date object.")
            return None

    def extract_month(date_obj):
        """Extracts the month from a given date object."""
        if isinstance(date_obj, datetime.date):
            return date_obj.month
        else:
            print("Invalid input. Please provide a date object.")
            return None

    def extract_day(date_obj):
        """Extracts the day of the month from a given date object."""
        if isinstance(date_obj, datetime.date):
            return date_obj.day
        else:
            print("Invalid input. Please provide a date object.")
            return None

    def extract_hour(time_obj):
        """Extracts the hour from a given time object."""
        if isinstance(time_obj, datetime.time):
            return time_obj.hour
        else:
            print("Invalid input. Please provide a time object.")
            return None

    def extract_minute(time_obj):
        """Extracts the minute from a given time object."""
        if isinstance(time_obj, datetime.time):
            return time_obj.minute
        else:
            print("Invalid input. Please provide a time object.")
            return None

    def extract_second(time_obj):
        """Extracts the second from a given time object."""
        if isinstance(time_obj, datetime.time):
            return time_obj.second
        else:
            print("Invalid input. Please provide a time object.")
            return None

    def date_diff(date1, date2, unit="days"):
    
        if not isinstance(date1, datetime.date) or not isinstance(date2, datetime.date):
            print("Invalid input. Please provide two date objects.")
            return None

        delta = date2 - date1
        if unit == "days":
            return delta.days
        elif unit == "months":
            years = delta.days // 365
            months = (delta.days - years * 365) // 30
            return years * 12 + months
        elif unit == "years":
            return delta.days // 365
        else:
            print("Invalid unit specified. Please use 'days', 'months', or 'years'.")
            return None

    def date_value(date_string, format="%Y-%m-%d"):
        return datetime.datetime.strptime(date_string, format).date().toordinal()

    