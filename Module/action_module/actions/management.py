import openpyxl as excel
from openpyxl.utils import  range_boundaries, column_index_from_string
from openpyxl.utils.cell import coordinate_from_string,get_column_letter
import xlsxwriter as xw
import pandas as pd
import spire.xls as xls
import re
from openpyxl.workbook.defined_name import DefinedName
from openpyxl.utils import quote_sheetname, absolute_coordinate
from openpyxl.workbook.defined_name import DefinedName
from openpyxl.utils import quote_sheetname, absolute_coordinate
class management:
  """
  Switch sheet: Navigate between different sheets within the workbook.
  Sort: Arrange the data within a range of cells in ascending or descending order.
  Filter: Display only the rows of data that meet specified criteria.
  Delete filter: Remove filter criteria from a range of cells.
  Slicer: Create a slicer to filter data interactively.
  Move rows: Change the position of rows within the spreadsheet.
  Move columns: Change the position of columns within the spreadsheet.
  Group: Group rows or columns together for organization.
  Ungroup: Remove grouping from rows or columns.
  Hide rows: Conceal rows from view.
  Hide columns: Conceal columns from view.
  Unhide rows: Reveal hidden rows.
  Unhide columns: Reveal hidden columns.
  Hide sheet: Make a sheet invisible within the workbook.
  Unhide sheet: Make a previously hidden sheet visible again.
  Set password: Protect the workbook or specific sheets with a password.
  Transpose: Swap rows and columns in a range of cells.
  Create named range: Define a named range of cells for easier reference.
  Delete named range: Remove a named range from the workbook.
  Data consolidation: Combine data from multiple ranges into a single range.
  Freeze panes: Lock rows or columns to keep them visible while scrolling.
  Unfreeze panes: Remove frozen panes from the spreadsheet.
  Split panes: Divide the worksheet window into multiple resizable panes
  """
  def __init__(self):
    return 


  @staticmethod
  def switch_sheet(workbook: excel.Workbook, sheet_name: str):
    workbook.active = workbook[sheet_name]
    return workbook
  
  @staticmethod
  def sort(workbook_path: str, source_sheet_name: str, key: str,output_workbook_name: str,ascending: bool = True):
    wb = pd.read_excel(workbook_path,source_sheet_name)
    if key in wb.columns:
      df = wb.sort_values(by=key, ascending=ascending)
    else:
      return 
    df.to_excel(output_workbook_name, source_sheet_name, index=False)
  @staticmethod
  def hide_null_value_rows(sheet, column, hidden):
    column_index = column_index_from_string(column)
    for row in sheet.iter_rows(min_row=2):
      cell = row[column_index - 1]
      if cell.value is None:
        sheet.row_dimensions[row[0].row].hidden = hidden
    return sheet
#===================== in progress ============================
#==============================================================  
  #filter_criteria looks like this filter_criteria = {"key":"operator value" or "key":[start_range, end_range] or "key":"value" for equality or "key":"value*" for wildcard autocomplete}
  @staticmethod
  def filter(workbook_path: str, source_sheet_name: str,  key: str = "name", criteria = [], output_workbook_path: str = None):
    # df = pd.read_feather(workbook_path,source_sheet_name)
    # for operator, value in criteria:
    #   if operator == ">":
    #     df = df[df[key] > value]
    #   elif operator == "<":
    #     df = df[df[key] < value]
    #   elif operator == "==":
    #     df = df[df[key] == value]
    #   elif operator == "<=":
    #     df = df[df[key] <= value]
    #   elif operator == ">=":
    #     df = df[df[key] >= value]
    #   elif operator == "!=":
    #     df = df[df[key] != value]
    #   elif operator == "contains":
    #     df = df[df[key].str.contains(value)]
    #   elif operator == "startswith":
    #     df = df[df[key].str.startswith(value)]
    #   elif operator == "endswith":
    #     df = df[df[key].str.endswith(value)]
    #   else:
    #     raise ValueError(f"Unsupported operator: {operator}")
    # df.to_excel(output_workbook_path)
    wb = excel.load_workbook(workbook_path)
    ws = wb[source_sheet_name]
    names = [name[0].value.lower() for name in ws.columns]
    # print(names)
    index = names.index(key.lower())
    key = get_column_letter(index + 1)
    print(index, key)
    ws.auto_filter.ref = f"{key}1:{key}{ws.max_row}"
    ws.auto_filter.add_filter_column(index + 1,vals=[criteria], blank = False)
    # ws = management.hide_null_value_rows(ws, key, True)
    wb.save(output_workbook_path)
    wb.close()
      
    
      
  @staticmethod
  def unfilter(workbook_path: str, source_sheet_name: str, output_workbook_path: str):
    workbook = excel.load_workbook(workbook_path)
    sheet = workbook[source_sheet_name]
    sheet.auto_filter = None
    workbook.save(output_workbook_path)
    workbook.close()    
    
  @staticmethod
  def slicer(workbook_path: str, source_sheet_name: str, source_range: str, key: str, value: str):
    return 
#==============================================================
  @staticmethod
  def move_rows(workbook_path: str, source_sheet_name: str, row_range: str, new_pos: str):
    workbook = excel.load_workbook(workbook_path)
    worksheet = workbook[source_sheet_name]
    old_position = range_boundaries(row_range)
    new_pos = range_boundaries(new_pos)
    end_pos = (new_pos[0] - old_position[0], new_pos[1] - old_position[1])
    row_range = row_range.split(":")
    rng = f"{row_range[0]}:{get_column_letter(worksheet.max_column)}{re.sub("[^0-9]",row_range[1])}"
    worksheet.move_range(rng, end_pos[0], end_pos[1])
    workbook.save(workbook_path)
    workbook.close()
    
  @staticmethod
  def move_columns_by_name(workbook_path: str, source_sheet_name: str, column_name: str, new_pos: str):
    workbook = excel.load_workbook(workbook_path)
    worksheet = workbook[source_sheet_name]
    names = [name.lower() for name in worksheet.columns]
    index = names.index(column_name.lower())
    key = get_column_letter(index)
    old_pos = range_boundaries(f"{key}1")
    new_pos = range_boundaries(new_pos)
    end_pos = (new_pos[0] - old_pos[0], new_pos[1] - old_pos[1])
    worksheet.move_range(f"{key}1:{key}{worksheet.max_row}", end_pos[0], end_pos[1])
    workbook.save(workbook_path)
    workbook.close()
  
  @staticmethod
  def move_columns_by_range(workbook_path: str, source_sheet_name: str, column_range: str, new_pos: str):
    workbook = excel.load_workbook(workbook_path)
    worksheet = workbook[source_sheet_name]
    old_position = range_boundaries(column_range)
    new_pos = range_boundaries(new_pos)
    end_pos = (new_pos[0] - old_position[0], new_pos[1] - old_position[1])
    column_range = column_range.split(":")
    rng = f"{column_range[0]}:{get_column_letter(worksheet.max_column)}{re.sub("[^0-9]",column_range[1])}"
    worksheet.move_range(rng, end_pos[0], end_pos[1])
    workbook.save(workbook_path)
    workbook.close()
    
  @staticmethod
  def group(workbook_path: str, source_sheet_name: str, source_range: str, by: str, output_workbook_path: str,hidden: bool = True):
    workbook = excel.load_workbook(workbook_path)
    worksheet = workbook[source_sheet_name]
    if source_range != None:
      start_loc, end_loc = source_range.split(":")
      start = range_boundaries(start_loc)
      end = range_boundaries(end_loc)
    elif source_range == None and by.lower() == "row":
      start_loc = 2
      end_loc = worksheet.max_row
    elif source_range == None:
      start_loc = 2
      end_loc = worksheet.max_column
    if by.lower() == "row":
      worksheet.row_dimensions.group(start[1], end[3], hidden=hidden)
    else:
      worksheet.column_dimensions.group(start[0], end_loc[2], hidden=hidden)
    workbook.save(output_workbook_path)
    workbook.close()
    
  @staticmethod
  def ungroup(workbook_path: str, source_sheet_name: str, source_range: str, by: str, output_workbook_path: str):
    wb = xls.Workbook()
    wb.LoadFromFile(workbook_path)
    sheet: xls.Worksheet = wb.Worksheets[source_sheet_name]
    if source_range != "None":
      start_loc, end_loc = source_range.split(":")
      start = range_boundaries(start_loc)
      end = range_boundaries(end_loc)
    elif source_range == None and by.lower() == "row":
      start_loc = 2
      end_loc = 0
    elif source_range == None:
      start_loc = 1
      end_loc = 0
    if by.lower() == "row":
      sheet.UngroupByRows(start[1], end[3])
    else:
      sheet.UngroupByColumns(start[0], end_loc[2])
    sheet.UngroupByColumns(4, 6)
    wb.SaveToFile(output_workbook_path, xls.ExcelVersion.Version2016)
    wb.Dispose()
  
  @staticmethod
  def hide_rows(workbook_path: str, source_sheet_name: str, source_range: str):
    workbook = excel.load_workbook(workbook_path)
    worksheet = workbook[source_sheet_name]
    for row in worksheet[source_range]:
      for cell in row:
        cell.hidden = True
    workbook.save(workbook_path)
    
  @staticmethod #needs adjustment
  def hide_columns(workbook_path: str, source_sheet_name: str, source_range: str):
    workbook = excel.load_workbook(workbook_path)
    worksheet = workbook[source_sheet_name]
    for row in worksheet[source_range]:
      for cell in row:
        cell.hidden = True
    workbook.save(workbook_path)
  
  @staticmethod 
  def unhide_rows(workbook_path: str, source_sheet_name: str, source_range: str):
    workbook = excel.load_workbook(workbook_path)
    worksheet = workbook[source_sheet_name]
    for row in worksheet[source_range]:
      for cell in row:
        cell.hidden = False
    workbook.save(workbook_path)
    
  @staticmethod #needs adjustment
  def unhide_columns(workbook_path: str, source_sheet_name: str, source_range: str):
    workbook = excel.load_workbook(workbook_path)
    worksheet = workbook[source_sheet_name]
    for row in worksheet[source_range]:
      for cell in row:
        cell.hidden = False
    workbook.save(workbook_path)
    
  @staticmethod
  def hide_sheet(workbook_path: str, source_sheet_name: str):
    workbook = excel.load_workbook(workbook_path)
    worksheet = workbook[source_sheet_name]
    worksheet.sheet_state = "hidden"
    workbook.save(workbook_path)
    
    
  @staticmethod
  def unhide_sheet(workbook_path: str, source_sheet_name: str):
    workbook = excel.load_workbook(workbook_path)
    worksheet = workbook[source_sheet_name]
    worksheet.sheet_state = "visible"
    workbook.save(workbook_path)
  
  @staticmethod
  def set_password(workbook_path: str, source_sheet_name: str, password: str):
    workbook = excel.load_workbook(workbook_path)
    if source_sheet_name not in workbook.sheetnames:  
      for sheet in workbook.worksheets:
        sheet.protection.set_password(password)
    else:
      worksheet = workbook[source_sheet_name]
      worksheet.protection.set_password(password)
    workbook.save(workbook_path)
  
  def transpose(workbook_path: str, source_sheet_name: str, source_range: str):
    workbook = excel.load_workbook(workbook_path)
    worksheet = workbook[source_sheet_name]
    min_col, min_row, max_col, max_row = range_boundaries(source_range)
    transposed_data = []
    for row in range(min_row, max_row + 1):
        transposed_row = []
        for col in range(min_col, max_col + 1):
            transposed_row.append(worksheet.cell(row=row, column=col).value)
        transposed_data.append(transposed_row)

    transposed_range = worksheet.iter_cols(min_row=min_row, min_col=min_col,
                                       max_row=max_row, max_col=max_col)
    for index, column in enumerate(transposed_range):
        for idx, cell in enumerate(column):
            cell.value = transposed_data[idx][index]
    workbook.save(workbook_path)
    
  @staticmethod
  def create_named_range(workbook_path: str, source_sheet_name: str, range_name: str, source_range: str, output_workbook_path: str):
    wb = excel.load_workbook(workbook_path)
    ws = wb[source_sheet_name]
    ref =  f"{quote_sheetname(ws.title)}!{absolute_coordinate(source_range)}"
    defn = DefinedName(range_name, attr_text=ref)
    wb.defined_names[range_name] = defn
    wb.save(output_workbook_path)
    wb.close()
    
    
  @staticmethod
  def delete_named_range(workbook_path: str, range_name: str, output_workbook_path: str):
    wb = excel.load_workbook(workbook_path)
    del wb.defined_names[range_name]
    wb.save(output_workbook_path)
    wb.close()

  @staticmethod
  def freeze_panes(workbook_path: str, sheet_name: str, top_left_cell: str, active_cell: str, output_workbook_path: str):
    workbook = excel.load_workbook(workbook_path)
    worksheet = workbook[sheet_name]
    worksheet.freeze_panes = worksheet.cell(top_left_cell, active_cell)
    workbook.save(output_workbook_path)
    workbook.close()

  @staticmethod
  def unfreeze_panes(workbook_path: str, sheet_name: str,  output_workbook_path: str):
    workbook = excel.load_workbook(workbook_path)
    worksheet = workbook[sheet_name]
    worksheet.freeze_panes = None
    workbook.save(output_workbook_path)
    workbook.close()
    
  @staticmethod
  def split_panes(workbook_path: str, sheet_name: str, vertical_split: str, horizontal_split: str, top_left_cell: str = 'A1', active_pane: str = "bottomRight"):
    wb = xw.Workbook(workbook_path)
    worksheet: xw.workbook.Worksheet = wb.get_worksheet_by_name(sheet_name)
    worksheet.split_panes(vertical_split*10, horizontal_split*8.42)
    wb.close()
    
  @staticmethod
  def data_consolidate(workbook_path: str, sheet_names: list[str], tgt_sheet: str = None):
    worksheets = []
    for sheet in sheet_names:
      worksheets.append(pd.read_excel(workbook_path, sheet_name= sheet))
    objs = pd.concat([worksheets], ignore_index = True)
    if tgt_sheet != None:
      with pd.ExcelWriter(workbook_path, engine='openpyxl', mode='a') as writer:
          try:
              objs.to_excel(writer, sheet_name = tgt_sheet, index=False, header=None)
          except PermissionError:
              print("Close the file in Excel and try again.")
      
    
    
    
# management.sort("./Modules/Action_Module/task_instructions.xlsx","Sheet1","No.","./Modules/Action_Module/task_instructions.xlsx",False)
# management.group("./Modules/Action_Module/task_instructions.xlsx","Sheet1","A1:B20","row","./Modules/Action_Module/task_instructions.xlsx",True)
# management.ungroup("./Modules/Action_Module/task_instructions.xlsx","Sheet1","A1:B20","row","./Modules/Action_Module/task_instructions.xlsx")
management.filter("./Modules/Action_Module/task_instructions.xlsx","Sheet1","Categories", "Formatting","./Modules/Action_Module/task_instructions.xlsx")