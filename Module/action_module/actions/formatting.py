import pandas as pd
#=================================
#openpyxl lib
from openpyxl import load_workbook
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.worksheet.page import PrintPageSetup,PageMargins
from openpyxl.utils import get_column_letter,range_boundaries
from openpyxl.utils.cell import coordinate_from_string
from openpyxl.styles import Font, Alignment, Border, Side, PatternFill, NamedStyle, Protection
from openpyxl.formatting.rule import CellIsRule
from openpyxl.drawing.image import Image
#=================================

class formatting(object):
    """
Format cells: Change the appearance of cells, such as font, color, alignment, etc.
Set data type: Specify the data type for a range of cells (e.g., text, number, date).
Delete format: Remove formatting from cells.
Change page layout: Adjust page settings such as orientation, margins, and print area.
Set border: Add borders to cells or ranges of cells.
Resize cells: Adjust the width and height of cells.
Conditional formatting: Apply formatting to cells based on specific conditions or criteria.
Lock and unlock: Protect cells from editing or unlock them for editing.
Protect: Protect the workbook or specific sheets from unauthorized changes.
Unprotect: Remove protection from the workbook or specific sheets.
Drop-down list: Create a drop-down list control in a cell.
Data validation: Specify restrictions or rules for the data entered into cells.
Display formulas: Show or hide formulas in cells.
Wrap text: Wrap text within cells to display long content.
Unwrap text: Remove text wrapping from cells.
Autofit: Automatically adjust the width of columns or height of rows to fit the content.
    """
    def __init__(self):
        super().__init__()
        return
    
    @staticmethod
    def format_cells(workbook_path,sheet_name, range_start, range_end, bold: bool = True,  italic: bool = True, color: str = "FF0000", horizontalAlignment: str = "center", verticalAlignment: str = "center", fillColor: str = "FFFF00", fillType: str = "solid", borderStyle: str = "thin", output_workbook_path=None):
        if output_workbook_path == None:
            output_workbook_path = workbook_path
        workbook = load_workbook(workbook_path)
        sheet = workbook[sheet_name]
        start_row, start_column = range_boundaries(range_start)
        end_row, end_column = range_boundaries(range_end)
        for row_idx in range(start_row, end_row):
            for col_idx in range(start_column, end_column):
                cell = sheet.cell(row = row_idx, column = col_idx)
                cell.font = Font(bold = bold, italic = italic, color = color)  # Bold, italic, red font color
                cell.alignment = Alignment(horizontal = horizontalAlignment, vertical = verticalAlignment)  # Center align content
                cell.border = Border(left = Side(style = borderStyle), right = Side(style = borderStyle), top = Side(style = borderStyle), bottom=Side(style = borderStyle))  # Thin border around cell
                cell.fill = PatternFill(start_color = fillColor, end_color = fillColor, fill_type = fillType)  # Yellow background color
        workbook.save(output_workbook_path)
        workbook.close()
        
    @staticmethod
    def set_data_type(workbook_path: str, sheet_name: str, range_start, range_end, datatype, output_workbook_path = None):
        if output_workbook_path == None:
            output_workbook_path = workbook_path
        workbook = load_workbook(workbook_path)
        sheet = workbook[sheet_name]
        start_row, start_column = range_boundaries(range_start)
        end_row, end_column = range_boundaries(range_end)
        for row in range(start_row, end_row + 1):
            for column in range(start_column, end_column + 1):
                cell = sheet.cell(row=row, column=column)
                if column == start_column + 1:  # Check if column is second column in the range
                    cell.number_format = datatype  # Set data type as integer
                else:
                    cell.number_format = "@"  # Set data type as text
        workbook.save(output_workbook_path)
        workbook.close()
        
    @staticmethod
    def remove_format(workbook_path, sheet_name, range_start, range_end, output_workbook_path = None):
        if output_workbook_path == None:
            output_workbook_path = workbook_path        
        workbook = load_workbook(workbook_path)
        sheet = workbook[sheet_name]
        start_row, start_column = range_boundaries(range_start)
        end_row, end_column = range_boundaries(range_end)
        for row in range(start_row, end_row + 1):
            for column in range(start_column, end_column + 1):
                cell = sheet.cell(row=row, column=column)
                cell.alignment = NamedStyle(name='Normal').alignment  # Reset alignment
                cell.font = NamedStyle(name='Normal').font  # Reset font
                cell.fill = NamedStyle(name='Normal').fill  # Reset fill
                cell.border = NamedStyle(name='Normal').border  # Reset border
                cell.number_format = "General"  # Reset number format
        workbook.save(output_workbook_path)
        workbook.close()
        
    @staticmethod
    def change_page_layout(workbook_path, sheet_name, orientation='portrait', paper_size='A4', margin_top=0.5, margin_right=0.5, margin_bottom=0.5, margin_left=0.5, print_area=None, output_workbook_path= None):
        if output_workbook_path == None:
            output_workbook_path = workbook_path        
        workbook = load_workbook(workbook_path)
        sheet = workbook[sheet_name]
        page_setup = PrintPageSetup(sheet)
        page_margins = PageMargins()
        page_setup.orientation = orientation
        page_setup.paperSize = paper_size
        page_margins.top = margin_top
        page_margins.right = margin_right
        page_margins.bottom = margin_bottom
        page_margins.left = margin_left
        if print_area:
            sheet.print_area = print_area
        workbook.save(output_workbook_path)
        workbook.close()
        
    @staticmethod
    def set_border(workbook_path, sheet_name, range_start, range_end, style: str, color: str, output_workbook_path = None):
        if output_workbook_path == None:
            output_workbook_path = workbook_path        
        workbook = load_workbook(workbook_path)
        sheet = workbook[sheet_name]
        start_row, start_column = coordinate_from_string(range_start)
        end_row, end_column = coordinate_from_string(range_end)
        thin_border = Border(left = Side(style = style, color = color), 
                             right = Side(style = style, color = color), 
                             top = Side(style = style, color = color), 
                             bottom = Side(style = style, color = color))
        for row in sheet.iter_rows(min_row=start_row, min_col=start_column, max_row=end_row, max_col=end_column):
            for cell in row:
                cell.border = thin_border
        workbook.save(output_workbook_path)
        workbook.close()
        
    @staticmethod
    def resize_cells(workbook_path, sheet_name, start_cell, end_cell, column_width=None, row_height=None, output_workbook_path= None):
        if output_workbook_path == None:
            output_workbook_path = workbook_path        
        workbook = load_workbook(workbook_path)
        sheet = workbook[sheet_name]
        start_row, start_column = coordinate_from_string(start_cell)
        end_row, end_column = coordinate_from_string(end_cell)
        if column_width is not None:
            for col in range(start_column, end_column + 1):
                sheet.column_dimensions[get_column_letter(col)].width = column_width
        if row_height is not None:
            for row in range(start_row, end_row + 1):
                sheet.row_dimensions[row].height = row_height
        workbook.save(output_workbook_path)
        workbook.close()
        
    @staticmethod
    def apply_conditional_formatting(workbook_path, sheet_name, range_string, condition, style, output_workbook_path = None):
        if output_workbook_path == None:
            output_workbook_path = workbook_path        
        workbook = load_workbook(workbook_path)
        sheet = workbook[sheet_name]
        rule = CellIsRule(operator=condition.operator, formula=condition.formula, stopIfTrue=True, style=style)
        sheet.conditional_formatting.add(range_string, rule)
        workbook.save(output_workbook_path)
        workbook.close()
        
    @staticmethod
    def lock_cells(workbook_path, sheet_name, range_string, lock, output_workbook_path = None):
        if output_workbook_path == None:
            output_workbook_path = workbook_path        
        workbook = load_workbook(workbook_path)
        sheet = workbook[sheet_name]
        for row in sheet[range_string]:
                row.protection = Protection(locked = lock)
        sheet.protection.sheet = True
        workbook.save(output_workbook_path)
        workbook.close()
    
    @staticmethod
    def unlock_cells(workbook_path, sheet_name, range_string, output_workbook_path = None):
        if output_workbook_path == None:
            output_workbook_path = workbook_path        
        workbook = load_workbook(workbook_path)
        sheet = workbook[sheet_name]
        for row in sheet[range_string]:
                row.protection = None
        sheet.protection.sheet = False
        workbook.save(output_workbook_path)
        workbook.close()

    @staticmethod
    def protect(workbook_path, sheet_name, password=None,output_workbook_path = None):
        if output_workbook_path == None:
            output_workbook_path = workbook_path        
        workbook = load_workbook(workbook_path)
        sheet = workbook[sheet_name]
        if password:
            sheet.protection.sheet = True
            sheet.protection.password = password
        else:
            sheet.protection.sheet = True
        workbook.save(output_workbook_path) 
        workbook.close()   
        
    @staticmethod
    def unprotect(workbook_path, sheet_name, output_workbook_path = None):
        if output_workbook_path == None:
            output_workbook_path = workbook_path        
        workbook = load_workbook(workbook_path)
        sheet = workbook[sheet_name]
        sheet.protection.sheet = None
        sheet.protection.password = None
        workbook.save(output_workbook_path)
        workbook.close()
        
    @staticmethod
    def dropdown_list(workbook_path, sheet_name, start_cell, end_cell, values, output_workbook_path = None):
        if output_workbook_path == None:
            output_workbook_path = workbook_path        
        workbook = load_workbook(workbook_path)
        sheet = workbook[sheet_name]
        data_validation = DataValidation(type="list", formula1='"' + ','.join(values) + '"', showDropDown=True)
        for row in sheet[start_cell:end_cell]:
            for cell in row:
                sheet.add_data_validation(data_validation)
                data_validation.add(cell)
        workbook.save(output_workbook_path)
        workbook.close()
        
    @staticmethod
    def set_border(workbook_path, sheet_name, range_start, range_end, wrap_text: bool = True,output_workbook_path = None):
        if output_workbook_path == None:
            output_workbook_path = workbook_path        
        workbook = load_workbook(workbook_path)
        sheet = workbook[sheet_name]
        start_row, start_column = coordinate_from_string(range_start)
        end_row, end_column = coordinate_from_string(range_end)
        for row in sheet.iter_rows(min_row=start_row, min_col=start_column, max_row=end_row, max_col=end_column):
            for cell in row:
                cell.alignment.wrap_text = wrap_text
        workbook.save(output_workbook_path) 
        workbook.close()
    
    @staticmethod
    def data_validation(workbook_path, sheet_name, range_start, range_end, validation_type, criteria, prompt=None, error_message=None,output_workbook_path = None):
        if output_workbook_path == None:
            output_workbook_path = workbook_path        
        workbook = load_workbook(workbook_path)
        sheet = workbook[sheet_name]
        data_validation = DataValidation(type=validation_type, formula1=criteria, promptTitle=prompt, prompt=prompt, errorTitle=error_message, error=error_message)
        sheet.add_data_validation(data_validation)
        data_validation.add(sheet[range_start:range_end])
        workbook.save(output_workbook_path)
        workbook.close()
        
    @staticmethod
    def insert_checkbox(workbook_path, sheet_name, cell_reference, output_workbook_path=None):
        if output_workbook_path == None:
            output_workbook_path = workbook_path 
        workbook = load_workbook(workbook_path)
        sheet = workbook[sheet_name]
        checkbox_image_path = "checkbox.png"
        img = Image(checkbox_image_path)
        img.anchor = cell_reference
        sheet.add_image(img)
        workbook.save(output_workbook_path)
        workbook.close()
        
    @staticmethod
    def display_formulas(workbook_path, sheet_name, output_workbook_path, display: bool = True):
        if output_workbook_path == None:
            output_workbook_path = workbook_path         
        workbook = load_workbook(workbook_path)
        sheet = workbook[sheet_name]
        sheet.sheet_view.showFormulas = display
        workbook.save(output_workbook_path)
        workbook.close()
        
    @staticmethod #revise later
    def wrap_text(workbook_path, sheet_name, range_start, range_end, wrapper,output_workbook_path):
        if output_workbook_path == None:
            output_workbook_path = workbook_path         
        workbook = load_workbook(workbook_path)
        sheet = workbook[sheet_name]
        for row in sheet[range_start:range_end]:
            row.alignment = Alignment(wrapText = wrapper)
        workbook.save(output_workbook_path)
        workbook.close()
        
    @staticmethod
    def add_data_validation(workbook_path, sheet_name, range_start, range_end, validation_type, criteria,output_workbook_path, prompt=None, error_message=None):
        if output_workbook_path == None:
            output_workbook_path = workbook_path         
        workbook = load_workbook(workbook_path)
        sheet = workbook[sheet_name]
        data_validation = DataValidation(type=validation_type, formula1=criteria, promptTitle=prompt, prompt=prompt, errorTitle=error_message, error=error_message)
        sheet.add_data_validation(data_validation)
        data_validation.add(sheet[range_start:range_end])
        workbook.save(output_workbook_path)
        workbook.close()
    
    def autofit_columns(workbook_path, sheet_name, output_workbook_path):
        if output_workbook_path == None:
            output_workbook_path = workbook_path         
        workbook = load_workbook(workbook_path)
        sheet = workbook[sheet_name]
        for column in sheet.columns:
            max_length = 0
            column_letter = get_column_letter(column[0].column)
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(cell.value)
                except:
                    pass
            adjusted_width = (max_length + 2) * 1.2 
            sheet.column_dimensions[column_letter].width = adjusted_width
        workbook.save(output_workbook_path)
        workbook.close()
        
        