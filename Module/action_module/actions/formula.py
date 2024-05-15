from openpyxl import load_workbook, worksheet, Workbook
import openpyxl as excel
from helper_functions import excel_column_to_index
import pandas as pd
from openpyxl.workbook.defined_name import DefinedName
from openpyxl.utils import get_column_letter, column_index_from_string,range_boundaries
from openpyxl.utils.cell import coordinate_from_string
import calendar
import datetime
def is_valid_formula(formula_str):
  # Implement logic to check for syntax errors (e.g., using regular expressions)
  # This is a basic example, you can improve it for robust validation
  return formula_str.startswith('=') and all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-*/()' for c in formula_str[1:])



class formula:
    """
Logical functions: Evaluate logical conditions and return true or false values.
    IF: Evaluates a specified condition and returns one value if the condition is true and another value if the condition is false.
    AND: Returns TRUE if all the supplied arguments evaluate to TRUE; otherwise, it returns FALSE.
    OR: Returns TRUE if any of the supplied arguments evaluate to TRUE; otherwise, it returns FALSE.
    NOT: Returns the opposite of a logical value; it converts TRUE to FALSE and FALSE to TRUE.
    TRUE: Returns the logical value TRUE.
    FALSE: Returns the logical value FALSE.
    IFERROR: Checks whether a formula results in an error and returns a specified value if an error is encountered; otherwise, it returns the result of the formula.
    IFNA: Checks whether a formula results in the #N/A error value and returns a specified value if #N/A is encountered; otherwise, it returns the result of the formula.
    IFS: Checks multiple conditions and returns a value that corresponds to the first TRUE condition; it replaces nested IF statements.
    SWITCH: Evaluates an expression against a list of values and returns a corresponding result; it replaces multiple IF statements.
    XOR: Returns TRUE if an odd number of arguments evaluate to TRUE; otherwise, it returns FALSE.
    ISEVEN: Checks whether a number is even and returns TRUE if the number is even; otherwise, it returns FALSE.
    ISODD: Checks whether a number is odd and returns TRUE if the number is odd; otherwise, it returns FALSE.
    IFERROR: Checks whether a formula results in an error and returns a specified value if an error is encountered; otherwise, it returns the result of the formula.
    IFNA: Checks whether a formula results in the #N/A error value and returns a specified value if #N/A is encountered; otherwise, it returns the result of the formula.
    ISEMPTY: Checks whether a specified cell is empty and returns TRUE if the cell is empty; otherwise, it returns FALSE.
    ISBLANK: Checks whether a specified cell is blank (i.e., contains no value) and returns TRUE if the cell is blank; otherwise, it returns FALSE.

    """
    def __init__(self):
        return
    
     
    #============================= date functions ===========================
    @staticmethod
    def update_cell_value(workbook_path, sheet_name, cell_range, new_value, output_workbook_path):
        wb = load_workbook(workbook_path)
        sheet = wb[sheet_name]
        position = range_boundaries(cell_range)
        for row in range(position[1], position[3] + 1):
            for col in range(position[0], position[2] + 1):
                sheet.cell(row,col).value = new_value
        wb.save(output_workbook_path)
        wb.close()
    
    @staticmethod
    def TODAY():
        return datetime.date.today()
    
    @staticmethod
    def NOW():
        return datetime.datetime.now()
    
    @staticmethod
    def DATE(date_str):
        return datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
    
    @staticmethod
    def TIME(time_str):
        return datetime.datetime.strptime(time_str, '%H:%M:%S').time()
    
    @staticmethod
    def YEAR(datetime_str):
        return datetime.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
    
    
    @staticmethod
    def MONTH(date_str):
        return datetime.datetime.strptime(date_str, '%Y-%m-%d').toordinal()
    
    @staticmethod
    def DAY(date_str):
        return datetime.datetime.strptime(date_str, '%Y-%m-%d').weekday()
    
    @staticmethod
    def HOUR(datetime_str):
        return datetime.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S').hour
    
    @staticmethod
    def MINUTE(datetime_str):
        return datetime.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S').minute
    
    @staticmethod
    def SECOND(datetime_str):
        return datetime.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S').second
    
    @staticmethod
    def DATEDIF(date_str_1, date_str_2, date_part):
        date_1 = datetime.datetime.strptime(date_str_1, '%Y-%m-%d %H:%M:%S') 
        date_2 =  datetime.datetime.strptime(date_str_2, '%Y-%m-%d %H:%M:%S')
        if date_1 > date_2:
            date_1, date_2 = date_2, date_1
        delta = date_2 - date_1
        if date_part == 'YEAR':
            return delta.days // 365
        elif date_part == 'MONTH':
            return delta.days // 30
        elif date_part == 'DAY':
            return delta.days
        elif date_part == 'HOUR':
            return delta.seconds 
        elif date_part == 'MINUTE':
            return delta.seconds // 60
        
    @staticmethod
    def DATEVALUE(date_str):
        return datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S').toordinal()
    
    @staticmethod
    def TIMEVALUE(time_str):
        return datetime.datetime.strptime(time_str, '%H:%M:%S').time()
    
    @staticmethod
    def EOMONTH(date_str):
        year, month = map(int, date_str.split('-'))
        _, days_in_month = calendar.monthrange(year, month)
        return datetime.date(year, month, days_in_month)
    
    @staticmethod
    def WORKDAY(date_str, N):
        return # Calculates the date of the Nth workday before or after a given date, excluding weekends and specified holidays.
    
    
    @staticmethod
    def NETWORKDAYS(date_str_1, date_str_2, holiday_list):
        return #Calculates the number of workdays between two dates, excluding weekends and specified holidays.
    
    #===================================================================================
    #============================= Logical functions ===================================
    
    
from xlwings import constants as win32c
from constants import constants
import win32com.client as win32
from typing import List
from openpyxl.utils import get_column_letter
import os

class Formula_App():
    def __init__(self, app = 'excel', api_doc = None) -> None:
        super().__init__()
        self.appName = app
        self.__excel = None
        self.__currentWB = None

        if api_doc is not None:
            for key, value in api_doc.items():
                if value.get('display') is not None:
                    setattr(self, value['display'], self.__getattribute__(key))

    @property
    def activeAPP(self):
        if not self.__excel:
            try:
                self.__excel = win32.Dispatch('Excel.Application') if self.appName == 'excel' else win32.Dispatch('ket.Application')
                self.__excel.DisplayAlerts = False
                self.__excel.Visible = False
            except:
                raise Exception('{} is not running.'.format(self.appName))
        return self.__excel
    
    @property
    def activeWB(self):
        if self.__currentWB is not None:
            return self.__currentWB
        return self.activeAPP.ActiveWorkbook
    
    @activeWB.setter
    def activeWB(self, wb):
        self.__currentWB = wb
    
    @property
    def activeWS(self):
        return self.activeWB.ActiveSheet
    
    def toRangeUsingSheet(self, source: str):
        if '!' in source:
            sheet_name, source = source.split('!')
            sheet_name = sheet_name.strip("'") 
            if sheet_name not in [sheet.Name for sheet in self.activeWB.Worksheets]:
                raise ValueError(f'Sheet {sheet_name} does not exist.')
            sheet = self.activeWB.Sheets(sheet_name)
        else:
            raise Exception('The range must contain a sheet name.')
        
        if source.isdigit():
            return sheet.Rows(source)
        elif source.isalpha():
            return sheet.Columns(source)
        elif ':' in source or source.isalnum():
            return sheet.Range(source)
        
    def toRange(self, sheet,  source: str):
        if source.isdigit():
            return sheet.Rows(source)
        elif source.isalpha():
            return sheet.Columns(source)
        elif ':' in source or source.isalnum():
            return sheet.Range(source)
    
    def OpenWorkbook(self, path: str) -> None:
        self.__currentWB = self.activeAPP.Workbooks.Open(os.path.abspath(path))

    def SaveWorkbook(self, path: str) -> None:
        self.activeWB.SaveAs(os.path.abspath(path))
    
    def closeWorkBook(self) -> None:
        self.activeWB.Close()
        
    @staticmethod
    def TODAY():
        return datetime.date.today()
    
    @staticmethod
    def NOW():
        return datetime.datetime.now()
    
    @staticmethod
    def DATE(date_str):
        return datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
    
    @staticmethod
    def TIME(time_str):
        return datetime.datetime.strptime(time_str, '%H:%M:%S').time()
    
    @staticmethod
    def YEAR(datetime_str):
        return datetime.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
    
    
    @staticmethod
    def MONTH(date_str):
        return datetime.datetime.strptime(date_str, '%Y-%m-%d').toordinal()
    
    @staticmethod
    def DAY(date_str):
        return datetime.datetime.strptime(date_str, '%Y-%m-%d').weekday()
    
    @staticmethod
    def HOUR(datetime_str):
        return datetime.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S').hour
    
    @staticmethod
    def MINUTE(datetime_str):
        return datetime.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S').minute
    
    @staticmethod
    def SECOND(datetime_str):
        return datetime.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S').second
    
    @staticmethod
    def DATEDIF(date_str_1, date_str_2, date_part):
        date_1 = datetime.datetime.strptime(date_str_1, '%Y-%m-%d %H:%M:%S') 
        date_2 =  datetime.datetime.strptime(date_str_2, '%Y-%m-%d %H:%M:%S')
        if date_1 > date_2:
            date_1, date_2 = date_2, date_1
        delta = date_2 - date_1
        if date_part == 'YEAR':
            return delta.days // 365
        elif date_part == 'MONTH':
            return delta.days // 30
        elif date_part == 'DAY':
            return delta.days
        elif date_part == 'HOUR':
            return delta.seconds 
        elif date_part == 'MINUTE':
            return delta.seconds // 60
        
    @staticmethod
    def DATEVALUE(date_str):
        return datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S').toordinal()
    
    @staticmethod
    def TIMEVALUE(time_str):
        return datetime.datetime.strptime(time_str, '%H:%M:%S').time()
    
    @staticmethod
    def EOMONTH(date_str):
        year, month = map(int, date_str.split('-'))
        _, days_in_month = calendar.monthrange(year, month)
        return datetime.date(year, month, days_in_month)
    
    @staticmethod
    def WORKDAY(date_str, N):
        return # Calculates the date of the Nth workday before or after a given date, excluding weekends and specified holidays.
    
    
    @staticmethod
    def NETWORKDAYS(date_str_1, date_str_2, holiday_list):
        return #
    
    
    #=================================================
    #============= logical functions =================
    #fix this
    def IF_(self, workbook_path: str, source_sheet: str, cell_range: str, condition: str, true_value: str, false_value: str, destination_sheet: str,target_range: str, output_workbook_path: str, element_wise: bool = True):
        self.OpenWorkbook(workbook_path)
        sheet = self.activeWB.Sheets(source_sheet)
        source_range_obj = self.toRange(sheet, cell_range)
        if destination_sheet != None:
            dest_sheet= self.activeWB.Sheets(destination_sheet)
            destination_range_obj = self.toRange(dest_sheet, target_range)
        else:
            destination_range_obj = self.toRange(source_sheet, target_range)
        if element_wise:
            for i, cell in enumerate(destination_range_obj):
                cell_formula = '=IF({destination_range_obj[i].Address}={source_range_obj[i].Value},{value_if_true_range_obj[i].Address},{value_if_false_range_obj[i].Address})'
                cell.Formula = cell_formula
        else:
            if_formula = '=IF({condition_range}={condition_range_obj[0].Value},{value_if_true_range}={value_if_true_range_obj[0].Value},{value_if_false_range}={value_if_false_range_obj[0].Value})'
            destination_range_obj.Formula = if_formula
        destination_range_obj.Calculate()
        self.SaveWorkbook(output_workbook_path)
        self.closeWorkBook()
        
    def AND_(self, workbook_path: str, source_sheet: str, cell_range: str, destination_sheet: str,target_range: str, output_workbook_path: str, element_wise: bool = True):
        self.OpenWorkbook(workbook_path)
        sheet = self.activeWB.Sheets(source_sheet)
        source_range_obj = self.toRange(sheet, cell_range)
        if destination_sheet != None:
            dest_sheet= self.activeWB.Sheets(destination_sheet)
            destination_range_obj = self.toRange(dest_sheet, target_range)
        else:
            destination_range_obj = self.toRange(source_sheet, target_range)
        if element_wise:
            for cell in destination_range_obj:
                cell_formula = f'=AND({cell.Address}={source_range_obj})'
                cell.Formula = cell_formula
        else:
            and_formula = f'=OR({source_sheet}!{cell_range})'
            destination_range_obj.Formula = and_formula
        destination_range_obj.Calculate()
        self.SaveWorkbook(output_workbook_path)
        self.closeWorkBook()
        
    def OR_(self, workbook_path: str, source_sheet: str, cell_range: str, destination_sheet: str,target_range: str, output_workbook_path: str, element_wise: bool = True):
        self.OpenWorkbook(workbook_path)
        sheet = self.activeWB.Sheets(source_sheet)
        source_range_obj = self.toRange(sheet, cell_range)
        if destination_sheet != None:
            dest_sheet= self.activeWB.Sheets(destination_sheet)
            destination_range_obj = self.toRange(dest_sheet, target_range)
        else:
            destination_range_obj = self.toRange(source_sheet, target_range)
        if element_wise:
            for cell in destination_range_obj:
                cell_formula = f'=OR({cell.Address}={source_range_obj})'
                cell.Formula = cell_formula
        else:
            and_formula = f'=OR({source_sheet}!{cell_range})'
            destination_range_obj.Formula = and_formula
        destination_range_obj.Calculate()
        self.SaveWorkbook(output_workbook_path)
        self.closeWorkBook()

    def XOR_(self, workbook_path: str, source_sheet: str, cell_range: str, destination_sheet: str,target_range: str, output_workbook_path: str, element_wise: bool = True):
        self.OpenWorkbook(workbook_path)
        sheet = self.activeWB.Sheets(source_sheet)
        source_range_obj = self.toRange(sheet, cell_range)
        if destination_sheet != None:
            dest_sheet= self.activeWB.Sheets(destination_sheet)
            destination_range_obj = self.toRange(dest_sheet, target_range)
        else:
            destination_range_obj = self.toRange(source_sheet, target_range)
        if element_wise:
            for cell in destination_range_obj:
                cell_formula = f'=XOR({cell.Address}={source_range_obj})'
                cell.Formula = cell_formula
        else:
            and_formula = f'=XOR({source_sheet}!{cell_range})'
            destination_range_obj.Formula = and_formula
        destination_range_obj.Calculate()
        self.SaveWorkbook(output_workbook_path)
        self.closeWorkBook()
        
    def NOT_(self, workbook_path: str, source_sheet: str, cell_range: str, destination_sheet: str,target_range: str, output_workbook_path: str, element_wise: bool = True):
        self.OpenWorkbook(workbook_path)
        sheet = self.activeWB.Sheets(source_sheet)
        source_range_obj = self.toRange(sheet, cell_range)
        if destination_sheet != None:
            dest_sheet= self.activeWB.Sheets(destination_sheet)
            destination_range_obj = self.toRange(dest_sheet, target_range)
        else:
            destination_range_obj = self.toRange(source_sheet, target_range)
        for cell in destination_range_obj:
            source_cell = source_range_obj.Cells[cell.Row - destination_range_obj.Row + 1, cell.Column - destination_range_obj.Column + 1]
            cell_formula = f'=NOT({source_sheet}!{source_cell.Address})'
            cell.Formula = cell_formula
        destination_range_obj.Calculate()
        self.SaveWorkbook(output_workbook_path)
        self.closeWorkBook()
        
    #finish remaining functions
    #========================================================================
    #==================== text functions ==============================

        
        
        
    
        