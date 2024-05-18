import pandas as pd
import xlsxwriter as xlsx
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
from openpyxl.pivot import table
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.drawing.image import Image
from openpyxl.chart import BarChart, LineChart, PieChart, AreaChart, Reference
#=================================

class pivot_table(object):
    def __init__(self)->None:
        super().__init__()
        return
    
    @staticmethod
    def create_pivot_table(workbook_path, source_sheet_name, source_range, indexes, output_workbook_path):
        """
        Create a pivot table in the specified destination sheet and cell.
        """
        # Load workbook and source sheet
        df = pd.read_excel(workbook_path, sheet_name=source_sheet_name)
        table = pd.pivot_table(df, values=source_range[0], index = indexes)
        writer = pd.ExcelWriter(output_workbook_path, engine='openpyxl',mode='a')
        df.to_excel(writer, sheet_name=source_sheet_name, index=False)
        
    @staticmethod
    def sort_pivot_table(workbook_path, pivot_table_name, field, ascending=True, output_workbook_path = None):
        df = pd.read_excel(workbook_path, sheet_name=pivot_table_name)
        df.sort_values(by=field, ascending=ascending, inplace=True)
        df.to_excel(output_workbook_path, sheet_name=pivot_table_name, index=False)
        
        
    @staticmethod
    def set_summary_type(workbook_path, src_sheet_name, field, summary_type):
        pass
                