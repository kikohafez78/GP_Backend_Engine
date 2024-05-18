from openpyxl import load_workbook
from openpyxl.drawing.image import Image
import openpyxl
from openpyxl.utils import get_column_letter
from openpyxl.utils.cell import coordinate_from_string, range_boundaries
from openpyxl.utils import cell as cs
import xlwings as xw

class entryandmanipulation(object):
    def __init__(self):
        super().__init__()
    
    @staticmethod
    def update_cell_value(workbook_path, sheet_name, cell_range, new_value, output_workbook_path):
        wb = load_workbook(workbook_path)
        sheet = wb[sheet_name]
        for row in sheet[cell_range]:
            for cell in row:
                cell.value = new_value
        wb.save(output_workbook_path)

    @staticmethod
    def delete_cells(workbook_path, sheet_name, cell_range,output_workbook_path):
        wb = load_workbook(workbook_path)
        sheet = wb[sheet_name]
        start_col, start_row, end_col, end_row = range_boundaries(cell_range)
        if start_col == end_col:
            sheet.delete_rows(start_row, end_row - start_row + 1)
        elif start_row == end_row:
            sheet.delete_cols(start_col, end_col - start_col + 1)
        else:
            sheet.delete_rows(start_row, end_row - start_row + 1)
            sheet.delete_cols(start_col, end_col - start_col + 1)
        wb.save(output_workbook_path)

    @staticmethod
    def merge_cells(workbook_path, sheet_name, cell_range, output_workbook_path):
        wb = load_workbook(workbook_path)
        sheet = wb[sheet_name]
        sheet.merge_cells(cell_range)
        wb.save(output_workbook_path)
        wb.close()

    @staticmethod
    def unmerge_cells(workbook_path, sheet_name, cell_range, output_workbook_path):
        wb = load_workbook(workbook_path)
        sheet = wb[sheet_name]
        sheet.unmerge_cells(cell_range)
        wb.save(output_workbook_path)
        wb.close()
    
    @staticmethod
    def split_text_to_columns(workbook_path, sheet_name, cell_range, output_workbook_path: str = "", delimiter:str = "."):
        wb = load_workbook(workbook_path)
        sheet = wb[sheet_name]
        start_col, start_row, end_col, end_row = range_boundaries(cell_range)
        max_len = 0
        for i in range(start_row, end_row + 1):
            for j in range(start_col, end_col + 1):
                cell = sheet.cell(row=i, column=j) 
                max_len = max(max_len, len(cell.value.split(delimiter)))
        ranges = f"{get_column_letter(end_col + 1)}1:{get_column_letter(sheet.max_column)}{sheet.max_row}"
        sheet.move_range(ranges,0,max_len - 1)
        for i in range(start_row, end_row + 1):
            for j in range(start_col, end_col + 1):
                cell = sheet.cell(row=i, column=j)
                string = cell.value.split(delimiter) 
                for k, value in enumerate(string):
                        sheet.cell(row=i, column=j + k).value = value
        wb.save(output_workbook_path)
        wb.close()

    @staticmethod
    def insert_row(workbook_path, sheet_name, row_number, output_workbook_path:str = ""):
        wb = load_workbook(workbook_path)
        sheet = wb[sheet_name]
        ranges = f"A{row_number + 1}:{get_column_letter(sheet.max_column)}{sheet.max_row}"
        sheet.move_range(ranges,0,1)
        sheet.insert_rows(row_number)
        wb.save(output_workbook_path)
        wb.close()

    @staticmethod
    def insert_column(workbook_path, sheet_name, column_number, column_name: str = "new column", output_workbook_path:str = ""):
        wb = load_workbook(workbook_path)
        sheet = wb[sheet_name]
        ranges = f"{get_column_letter(column_number)}1:{get_column_letter(sheet.max_column)}{sheet.max_row}"
        sheet.move_range(ranges,0,1)
        sheet.insert_cols(column_number)
        sheet[get_column_letter(column_number) + "1"].value = column_name
        wb.save(output_workbook_path)
        wb.close()

    @staticmethod
    def autofill(workbook_path, sheet_name, cell_range, output_workbook_path):
        app = xw.App(visible = False)
        workbook  = app.books.open(workbook_path)
        sheet = workbook.sheets[sheet_name]
        rng = sheet.range(cell_range)
        rng.api.Autofill(rng.api, 0)
        workbook.save(output_workbook_path)
        workbook.close()
        app.quit()


    @staticmethod
    def copy_paste_range(source_workbook_path: str, target_workbook_path: str,source_sheet_name: str, target_sheet_name: str, source_range: str, target_cell: str):
        src_wb = load_workbook(source_workbook_path)
        tgt_wb = load_workbook(target_workbook_path)
        src_sheet = src_wb[source_sheet_name]
        tgt_sheet = tgt_wb[target_sheet_name]
        start_column, start_row, end_column, end_row = range_boundaries(source_range)
        position = range_boundaries(target_cell)
        print(start_column,start_row,end_column,end_row)
        print(position)
        for i in range(start_row, end_row + 1):
            for j in range(start_column, end_column + 1):
                cell = src_sheet.cell(row=i, column=j)
                tgt_sheet.cell(row = position[1] + i - start_row, column = position[0] + j - start_column).value = cell.value
        tgt_wb.save(target_workbook_path)
        tgt_wb.close()
        src_wb.close()

    @staticmethod
    def copy_paste_format(source_workbook_path: str, target_workbook_path: str,source_sheet_name: str, target_sheet_name: str, source_range: str, target_cell: str):
        src_wb = load_workbook(source_workbook_path)
        tgt_wb = load_workbook(target_workbook_path)
        src_sheet = src_wb[source_sheet_name]
        tgt_sheet = tgt_wb[target_sheet_name]
        start_column, start_row, end_column, end_row = range_boundaries(source_range)
        position = range_boundaries(target_cell)
        print(start_column,start_row,end_column,end_row)
        print(position)
        for i in range(start_row, end_row + 1):
            for j in range(start_column, end_column + 1):
                cell = src_sheet.cell(row=i, column=j)
                tgt_sheet.cell(row = position[1] + i - start_row, column = position[0] + j - start_column).value = cell.value
                if cell.has_style:
                    tgt_sheet.cell(row = position[1] + i - start_row, column = position[0] + j - start_column).style = cell.style    
        tgt_wb.save(target_workbook_path)
        tgt_wb.close()
        src_wb.close()

    @staticmethod
    def copy_sheet(source_wb_file, dest_wb_file, sheet_name, new_sheet_name="new Sheet"):
        src_wb = load_workbook(source_wb_file)
        tgt_wb = load_workbook(dest_wb_file)
        src_sheet = src_wb[sheet_name]
        tgt_wb.create_sheet(new_sheet_name)
        entryandmanipulation.copy_paste_format(source_wb_file, dest_wb_file,sheet_name,new_sheet_name,f"A1:{get_column_letter(src_sheet.max_column)}{src_sheet.max_row}","A1")
        src_wb.close()
        tgt_wb.close()


    @staticmethod
    def cut_paste_range(src_workbook_path, tgt_workbook_path, src_sheet_name,tgt_sheet_name, source_range, target_range):
        entryandmanipulation.copy_paste_range(src_workbook_path,tgt_workbook_path, src_sheet_name, tgt_sheet_name,  source_range, target_range)
        entryandmanipulation.delete_cells(src_workbook_path, src_sheet_name, source_range)
        
        
    @staticmethod
    def find_and_replace(workbook_path, sheet_name, find_text, replace_text):
        wb = load_workbook(workbook_path)
        sheet = wb[sheet_name]
        if type(find_text) == str:
            for row in sheet:
                for cell in row:
                    if find_text in cell.value:
                        cell.value.replace(find_text, replace_text)
        else:
            for row in sheet:
                for cell in row:
                    if find_text == cell.value:
                        cell.value = replace_text
        wb.save(workbook_path)
        wb.close()

    @staticmethod
    def set_hyperlink(workbook_path, sheet_name, cell_range, url):
        wb = load_workbook(workbook_path)
        sheet = wb[sheet_name]
        for row in sheet[cell_range]:
            for cell in row:
                cell.hyperlink = url
        wb.save(workbook_path)

    @staticmethod
    def delete_hyperlink(workbook_path, sheet_name, cell_range):
        wb = load_workbook(workbook_path)
        sheet = wb[sheet_name]
        for row in sheet[cell_range]:
            for cell in row:
                if cell.hyperlink:
                    cell.hyperlink = None
        wb.save(workbook_path)

    @staticmethod
    def remove_duplicates(workbook_path, sheet_name, column_number):
        wb = load_workbook(workbook_path)
        sheet = wb[sheet_name]
        values = set()
        position = get_column_letter(column_number)
        for row in sheet[f"{position}2:{position}{sheet.max_row}"]:
            for cell in row:
                if cell.value in values:
                    sheet.delete_rows(cell.row)
                else:
                    values.add(cell.value)
        wb.save(workbook_path)

    @staticmethod
    def rename_sheet(workbook_path, old_sheet_name, new_sheet_name):
        wb = load_workbook(workbook_path)
        wb[old_sheet_name].title = new_sheet_name
        wb.save(workbook_path)

    @staticmethod
    def insert_checkbox(workbook_path, sheet_name, cell_range):
        wb = load_workbook(workbook_path)
        sheet = wb[sheet_name]
        sheet.cell(*cell_range).value = '☑'
        wb.save(workbook_path)
        wb.close()

    @staticmethod #needs work
    def insert_textbox(workbook_path, sheet_name, cell_range, text):
        wb = load_workbook(workbook_path)
        sheet = wb[sheet_name]
        img = Image('path/to/your/textbox_image.png')
        img.anchor = cell_range
        sheet.add_image(img)
        wb.save(workbook_path)
        wb.close()

    @staticmethod
    def create_sheet(workbook_path, sheet_name):
        wb = load_workbook(workbook_path)
        wb.create_sheet(title = sheet_name)
        wb.save(workbook_path)
        wb.close()

    @staticmethod
    def delete_sheet(workbook_path, sheet_name):
        wb = load_workbook(workbook_path)
        del wb[sheet_name]
        wb.save(workbook_path)
        wb.close()

    @staticmethod
    def clear_range(workbook_path, sheet_name, cell_range):
        wb = load_workbook(workbook_path)
        sheet = wb[sheet_name]
        for row in sheet[cell_range]:
            for cell in row:
                cell.value = None  # Clear cell value
                # if cell.has_style:
                #     cell.style = None# Remove any formatting
        wb.save(workbook_path)
        wb.close()






#testing
# entryandmanipulation.update_cell_value("./Modules/Action_Module/testing.xlsx","Sheet1","A1:B20","hello.world.kiko","./Modules/Action_Module/testing.xlsx")
# entryandmanipulation.delete_cells("./Modules/Action_Module/testing.xlsx","Sheet1","A1:B20","./Modules/Action_Module/testing.xlsx")
# entryandmanipulation.merge_cells("./Modules/Action_Module/testing.xlsx","Sheet1","A1:B10","./Modules/Action_Module/testing.xlsx")
# entryandmanipulation.split_text_to_columns("./Modules/Action_Module/testing.xlsx","Sheet1","A1:A20","./Modules/Action_Module/testing.xlsx")
# entryandmanipulation.autofill("./Modules/Action_Module/testing.xlsx", "Sheet1","A1:A20","./Modules/Action_Module/testing.xlsx")
# entryandmanipulation.insert_column("./Modules/Action_Module/seed_tasks.xlsx","Sheet1",2,"cash","./Modules/Action_Module/seed_tasks.xlsx")
# entryandmanipulation.insert_row("./Modules/Action_Module/seed_tasks.xlsx","Sheet1",2,"./Modules/Action_Module/seed_tasks.xlsx")
# entryandmanipulation.copy_paste_format("./Modules/Action_Module/task_instructions.xlsx","./Modules/Action_Module/task_instructions.xlsx","Sheet1","Sheet1","A1:A20","B1")
# entryandmanipulation.copy_paste_range("./Modules/Action_Module/task_instructions.xlsx","./Modules/Action_Module/task_instructions.xlsx","Sheet1","Sheet1","A1:A20","B1")
# entryandmanipulation.remove_duplicates("./Modules/Action_Module/task_instructions.xlsx","Sheet1", 6)
# entryandmanipulation.delete_hyperlink("./Modules/Action_Module/task_instructions.xlsx","Sheet1","A1:A20")
# entryandmanipulation.create_sheet("./Modules/Action_Module/task_instructions.xlsx","Sheet2")
# entryandmanipulation.delete_sheet("./Modules/Action_Module/task_instructions.xlsx","Sheet2")
# entryandmanipulation.find_and_replace("./Modules/Action_Module/task_instructions.xlsx","Sheet1",80,180)
# entryandmanipulation.clear_range("./Modules/Action_Module/task_instructions.xlsx","Sheet1","A1:A20")
