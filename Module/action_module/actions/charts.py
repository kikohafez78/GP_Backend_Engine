import pandas as pd
import xlsxwriter as chartcontrol
#=================================
#openpyxl lib
from openpyxl import load_workbook, worksheet, Workbook
from openpyxl.workbook.defined_name import DefinedName
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.worksheet.page import PrintPageSetup,PageMargins
from openpyxl.utils import get_column_letter, column_index_from_string,range_boundaries
from openpyxl.utils.cell import coordinate_from_string
from openpyxl.styles import Font, Alignment, Border, Side, PatternFill, NamedStyle, Protection
from openpyxl.formatting.rule import CellIsRule
from openpyxl.drawing.image import Image
from openpyxl.chart import BarChart, LineChart, PieChart, AreaChart, Reference
#=================================
#aesposecells lib
from asposecells.api import ProtectionType as pt, Workbook as wbs

class charts(object):
    def __init__(self)->None:
        super().__init__()
        return
    @staticmethod
    def create_chart(workbook_path, sheet_name, chart_type, data_range, chart_title, chart_location):
        workbook = load_workbook(workbook_path)
        sheet = workbook[sheet_name]
        if chart_type.lower() == 'bar':
            chart = BarChart()
        elif chart_type.lower() == 'line':
            chart = LineChart()
        elif chart_type.lower() == 'pie':
            chart = PieChart()
        elif chart_type.lower() == 'area':
            chart = AreaChart()
        else:
            raise ValueError("Invalid chart type. Supported types are 'bar', 'line', and 'pie'.")
        chart.title = chart_title
        min_row, min_col, max_row, max_col = data_range
        chart_data = Reference(sheet, min_col, min_row, max_col, max_row)
        chart_categories = Reference(sheet, min_col=1, min_row=2, max_row=6)
        chart.add_data(chart_data)
        chart.set_categories(chart_categories)
        sheet.add_chart(chart, chart_location)
        workbook.save(workbook_path)
        
    @staticmethod
    def create_pivot_table(workbook_path, sheet_name, pivot_sheet_name, data_range, pivot_index, pivot_columns, pivot_values):
        df = pd.read_excel(workbook_path, sheet_name=sheet_name, usecols=data_range)
        pivot_table = pd.pivot_table(df, index=pivot_index, columns=pivot_columns, values=pivot_values)
        with pd.ExcelWriter(workbook_path, engine='openpyxl', mode='a') as writer:
            pivot_table.to_excel(writer, sheet_name=pivot_sheet_name)
    
    @staticmethod
    def set_chart_name(workbook_path, sheet_name, old_chart_name, new_chart_name):
        workbook = load_workbook(workbook_path)
        sheet = workbook[sheet_name]
        
        