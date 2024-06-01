# import os
# import glob2 as gl # type: ignore
# import win32com.client as win32 #type: ignore
# from xlwings import constants as win32c
# import json
# from charts_types import constants as chart_classifier
# class excel_information_extractor:
#     def __init__(self, visible: bool, name: str):
#         self.name = name
#         self.app: win32.CDispatch = win32.Dispatch("Excel.Application")
#         self.app.Visible = visible 
    
#     def workspace_scan(self, path: str):
#         data = {}
#         files = gl.glob(path)
#         for file in files:
#             if file.endswith(".xlsx") or file.endswith(".csv"):
#                 name = file
#                 name.strip("csv")
#                 name.strip(".xlsx")
#                 data[name] = self.file_scan_(path, file)
    
#     def get_merged_range(self, cell, row, col, worksheet):
#         if cell.MergeCells:
#           # Get the top-left corner address (assuming it's the starting cell)
#           start_row = cell.Row
#           start_col = cell.Column

#           # Loop through subsequent cells in the same row to find the end of the merged range
#           merged_range_end_col = col
#           while merged_range_end_col < worksheet.UsedRange.Columns.Count and \
#               worksheet.Cells(start_row, merged_range_end_col + 1).MergeCells:
#             merged_range_end_col += 1

#           # Loop through subsequent rows to find the end of the merged range
#           merged_range_end_row = start_row
#           while merged_range_end_row < worksheet.UsedRange.Rows.Count and \
#               worksheet.Cells(merged_range_end_row + 1, start_col).MergeCells:
#             merged_range_end_row += 1

#           # Create a dictionary with merged range information
#           return {
#               "start_row": start_row,
#               "start_col": start_col,
#               "end_row": merged_range_end_row,
#               "end_col": merged_range_end_col,
#               "range": f"{worksheet.Cells(start_row, start_col).Address}:{worksheet.Cells(merged_range_end_row, merged_range_end_col).Address}"
#           }
#     def group_files_scan(self, paths: list[str], names: list[str]):
#         data = {}
#         for name, path in zip(names,paths):
#             data[name] = self.file_scan_(path, name)
#         return data
    
#     def same_workspace_scan(self, path: str, names: list[str]):
#         data = {}
#         for name in names:
#             data[name] = self.file_scan_(path, name)
#         return data
    
#     def file_scan_(self, workspace: str, file: str):
#         file_name = os.path.join(workspace, file)
        
#         if len(gl.glob(file_name)) == 0:
#             return "file not found"
        
#         if file.endswith(".xlsx") or file.endswith(".csv"):
#             workbook_info = {}
#             try:
#                 wb = self.app.Workbooks.Open(os.path.abspath(file_name))
#                 workbook_info["named_ranges"] = {}
#             except:    
#                 return "error with file found"
#             for name in wb.Names:
#                 workbook_info["named_ranges"][name.Name] = {
#                     "name": name.Name,
#                     "refers_to": str(name.RefersTo),  # Convert reference to string
#                 }
#             for sheet in wb.Sheets:
#                 workbook_info[sheet.Name] = {}
#                 workbook_info[sheet.Name]["columns"] = {}
#                 used_range = sheet.UsedRange

#                 # Assuming headers are in the first row
#                 first_row = used_range.Rows(1)

#                 # Loop through columns in the first row and extract information
#                 for col in range(1, first_row.Columns.Count + 1):
#                     cell_value = first_row.Cells(col).Value
#                     if cell_value:  # Check if cell has a value (skip empty cells)
#                         workbook_info[sheet.Name]["columns"][cell_value] = ({
#                         "name": cell_value,
#                         "data_type": sheet.Cells(2, col).NumberFormatLocal,  # Data type based on number format
#                         "formatting": sheet.Cells(2, col).NumberFormatLocal,  # Formatting code
#                         "start_position": sheet.Cells(1, col).Column,
#                         "end_position": sheet.Cells(used_range.Rows.Count, col).Column,
#                         })
#                 # print(workbook_info)
#                 workbook_info[sheet.Name]["charts"] = {}
#                 for chart_object in sheet.ChartObjects():
#                     chart = chart_object.Chart
#                     legend_info = {}
#                     if chart.HasLegend:
#                         # Extract legend entries (assuming text and font properties)
#                         for legend_entry in chart.Legend.LegendEntries():
#                             legend_info[legend_entry.Index] = {
#                                 "name": legend_entry.Index,
#                                 "font": {
#                                 "font_size": legend_entry.Font.Size,
#                                 "bold": legend_entry.Font.Bold,
#                                 "italic": legend_entry.Font.Italic,
#                                 "underline": legend_entry.Font.Underline,
#                                 },
#                             }
#                     workbook_info[sheet.Name]["charts"][chart.ChartTitle.Text] = {
#                         "name": chart.Name,
#                         "title": chart.ChartTitle.Text if chart.HasTitle else "",  # Check for title existence
#                         "type": chart_classifier.chart_types[str(chart.ChartType)],  # Numerical representation of chart type (refer to Excel documentation)
#                         "position": chart_object.Left,  # Left position of the chart object
#                         "size": (chart_object.Width, chart_object.Height),  # Width and height of the chart object
#                         "has_legend": chart.HasLegend,
#                         "data_source": "",
#                         "axes":[],
#                         "legend": legend_info if len(legend_info) > 0 else None# Boolean indicating presence of a legend
#                     } 
#                     # print(workbook_info)
#                     try:
#                     # Attempt using chart.ChartArea.LinkedSourceName (may not be available in all versions)
#                         workbook_info[sheet.Name]["charts"][chart.ChartTitle.Text][-1]["data_source"] = chart.ChartArea.LinkedSourceName
#                     except:
#                         pass 

#                     try:
#                     # Attempt using chart.ChartArea.ChartGroups(1).SourceName (may not be available in all versions)
#                         workbook_info[sheet.Name]["charts"][chart.ChartTitle.Text][-1]["data_source"] = chart.ChartArea.ChartGroups(1).SourceName  # Assuming first chart group
#                     except:
#                         pass

#                     # Extract axis information (assuming two axes - X and Y)
#                     for axis_id in range(0, 2):
#                         axis = chart.Axes(axis_id + 1)
#                         if axis.HasTitle:
#                             axis_info = {
#                                 "axis_type": axis.AxisType,
#                                 "title": axis.AxisTitle.Text,
#                             }
#                             workbook_info[sheet.Name]["charts"][chart.ChartTitle.Text]["axes"].append(axis_info)
#                 workbook_info[sheet.Name]["pivot_tables"] = {}
#                 for pivot_table in sheet.PivotTables():
#                     workbook_info[sheet.Name]["pivot_tables"][pivot_table.Name] = {
#                         "name": pivot_table.Name,
#                         "data_source": pivot_table.CacheSource if hasattr(pivot_table, 'CacheSource') else None,  # Data source name
#                         "row_fields": [],  # Placeholder for row fields
#                         "column_fields": [],  # Placeholder for column fields
#                         "value_fields": [],  # Placeholder for value fields
#                         "filters": [],  # Placeholder for filters (optional)
#                         "summary_types": [],  # Placeholder for summary types
#                     }
#                     # Extract row fields
#                     for field in pivot_table.PivotFields():
#                         # print(pivot_table.PivotFields(str(field)).Orientation)
#                         if pivot_table.PivotFields(str(field)).Orientation == win32c.PivotFieldOrientation.xlRowField:  # Check for row field type
#                             workbook_info[sheet.Name]["pivot_tables"][pivot_table.Name]["row_fields"].append(pivot_table.PivotFields(str(field)).Name)
#                         elif pivot_table.PivotFields(str(field)).Orientation == win32c.PivotFieldOrientation.xlColumnField:  # Check for column field type
#                             workbook_info[sheet.Name]["pivot_tables"][pivot_table.Name]["column_fields"].append(pivot_table.PivotFields(str(field)).Name)
#                         else:  # Check for value field type
#                             workbook_info[sheet.Name]["pivot_tables"][pivot_table.Name]["value_fields"].append(pivot_table.PivotFields(str(field)).Name)
#                             # workbook_info[sheet.Name]["pivot_tables"][pivot_table.Name]["summary_types"].append(str(pivot_table.PivotFields(str(field)).Function))  # Summary type

#                 # workbook_info[sheet.Name]["tables"] = {}
#                 # for used_range in sheet.UsedRange.Areas:
#                 #     if used_range.HasTableStyle:
#                 #         # Extract information (assuming formatting conventions)
#                 #         workbook_info[sheet.Name]["tables"][used_range.Name] = {
#                 #         "name": used_range.Name,  # Might not be reliable for tables
#                 #         "data_source": used_range.Address,  # Approximate data source range
#                 #         "has_header_row": used_range.Cells(1, 1).HasStyle,  # Check for header style (replace with your condition)
#                 #         "column_names": [],  # Placeholder for column names
#                 #         "row_count": used_range.Rows.Count - (1 if used_range.HasStyle else 0),  # Adjust for header
#                 #         "column_count": used_range.Columns.Count,
#                 #         }

#                 #         # Extract column names (assuming header row)
#                 #         if used_range.HasStyle:  # Adjust condition based on your header identification
#                 #             for col in range(1, used_range.Columns.Count + 1):
#                 #  
#                 # workbook_info[sheet.Name]["tables"][used_range.Name]["column_names"].append(used_range.Cells(1, col).Value)
#                 workbook_info[sheet.Name]["formulas"] = {}
#                 for cell in sheet.UsedRange:
#                     if cell.HasFormula:  # Check if cell contains a formula
#                         workbook_info[sheet.Name]["formulas"][cell.Formula] = {
#                         "cell": cell.Address,
#                         "formula": cell.Formula,  # Formula string
#                         "source_ranges": [],  # Placeholder for referenced ranges
#                         "destination_range": cell.Address,  # Destination (cell itself)
#                         }

#                         # Extract referenced ranges (may require parsing the formula string)
#                         for argument_index in range(cell.FormulaR1C1.Arguments.Count):
#                             argument = cell.FormulaR1C1.Arguments(argument_index)
#                         if argument.IsRange:  # Check if argument is a range reference
#                             workbook_info[sheet.Name]["formulas"][cell.Formula]["source_ranges"].append(argument.Text)
#                 workbook_info[sheet.Name]["merged_cells"] = []      
#                 for row in range(1, sheet.UsedRange.Rows.Count + 1):
#                     for col in range(1, sheet.UsedRange.Columns.Count + 1):
#                         cell = sheet.Cells(row, col)
#                         if cell.MergeCells:
#                             workbook_info[sheet.Name]["merged_cells"].append((cell.Address, cell.Value))
#                 # workbook_info[sheet.Name]["filters"] = []
#                 # if sheet.Filters.Count > 0:
#                 #     for filter in sheet.Filters:
#                 #         workbook_info[sheet.Name]["filters"].append({
#                 #         "name": filter.Name,
#                 #         "range": filter.Range.Address,  # Gets the affected data range address
#                 #         "criteria": str(filter.Criteria1)  # Might need adjustments for complex criteria
#                 #         })
#             wb.Close(SaveChanges = True)
                
                
                
#     def _destruct(self):
#         self.app.Quit()    

import os
import glob2 as gl # type: ignore
import win32com.client as win32 #type: ignore
from xlwings import constants as win32c
import json
from .charts_types import constants as chart_classifier
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

class excel_information_extractor:
    def __init__(self, visible: bool, name: str):
        self.name = name
        self.app: win32.CDispatch = win32.Dispatch("Excel.Application")
        self.app.Visible = visible 
    
    def workspace_scan(self, path: str):
        data = {}
        files = gl.glob(path)
        for file in files:
            if file.endswith(".xlsx") or file.endswith(".csv"):
                name = file
                name.strip("csv")
                name.strip(".xlsx")
                data[name] = self.file_scan_(path, file)
    
    def get_merged_range(self, cell, row, col, worksheet):
        if cell.MergeCells:
          # Get the top-left corner address (assuming it's the starting cell)
          start_row = cell.Row
          start_col = cell.Column

          # Loop through subsequent cells in the same row to find the end of the merged range
          merged_range_end_col = col
          while merged_range_end_col < worksheet.UsedRange.Columns.Count and \
              worksheet.Cells(start_row, merged_range_end_col + 1).MergeCells:
            merged_range_end_col += 1

          # Loop through subsequent rows to find the end of the merged range
          merged_range_end_row = start_row
          while merged_range_end_row < worksheet.UsedRange.Rows.Count and \
              worksheet.Cells(merged_range_end_row + 1, start_col).MergeCells:
            merged_range_end_row += 1

          # Create a dictionary with merged range information
          return {
              "start_row": start_row,
              "start_col": start_col,
              "end_row": merged_range_end_row,
              "end_col": merged_range_end_col,
              "range": f"{worksheet.Cells(start_row, start_col).Address}:{worksheet.Cells(merged_range_end_row, merged_range_end_col).Address}"
          }
    def group_files_scan(self, paths: list[str], names: list[str]):
        data = {}
        for name, path in zip(names,paths):
            data[name] = self.file_scan_(path, name)
        return data
    
    def same_workspace_scan(self, path: str, names: list[str]):
        data = {}
        for name in names:
            data[name] = self.file_scan_(path, name)
        return data
    
    def file_scan_(self, workspace: str, file: str):
        file_name = os.path.join(workspace, file)
        
        if len(gl.glob(file_name)) == 0:
            return "file not found"
        
        if file.endswith(".xlsx") or file.endswith(".csv"):
            workbook_info = {}
            try:
                wb = self.app.Workbooks.Open(os.path.abspath(file_name))
                workbook_info["named_ranges"] = {}
            except:    
                return "error with file found"
            for name in wb.Names:
                workbook_info["named_ranges"][name.Name] = {
                    "name": name.Name,
                    "refers_to": str(name.RefersTo),  # Convert reference to string
                }
            for sheet in wb.Sheets:
                workbook_info[sheet.Name] = {}
                workbook_info[sheet.Name]["columns"] = {}
                used_range = sheet.UsedRange

                # Assuming headers are in the first row
                first_row = used_range.Rows(1)

                # Loop through columns in the first row and extract information
                for col in range(1, first_row.Columns.Count + 1):
                    cell_value = first_row.Cells(col).Value
                    if cell_value:  # Check if cell has a value (skip empty cells)
                        workbook_info[sheet.Name]["columns"][cell_value] = ({
                        "name": cell_value,
                        "data_type": sheet.Cells(2, col).NumberFormatLocal,  # Data type based on number format
                        "formatting": sheet.Cells(2, col).NumberFormatLocal,  # Formatting code
                        "start_position": sheet.Cells(1, col).Column,
                        "end_position": sheet.Cells(used_range.Rows.Count, col).Column,
                        })
                # print(workbook_info)
                workbook_info[sheet.Name]["charts"] = {}
                for chart_object in sheet.ChartObjects():
                    chart = chart_object.Chart
                    legend_info = {}
                    if chart.HasLegend:
                        # Extract legend entries (assuming text and font properties)
                        for legend_entry in chart.Legend.LegendEntries():
                            legend_info[legend_entry.Index] = {
                                "name": legend_entry.Index,
                                "font": {
                                "font_size": legend_entry.Font.Size,
                                "bold": legend_entry.Font.Bold,
                                "italic": legend_entry.Font.Italic,
                                "underline": legend_entry.Font.Underline,
                                },
                            }
                    workbook_info[sheet.Name]["charts"][chart.ChartTitle.Text] = {
                        "name": chart.Name,
                        "title": chart.ChartTitle.Text if chart.HasTitle else "",  # Check for title existence
                        "type": chart_classifier.chart_types[str(chart.ChartType)],  # Numerical representation of chart type (refer to Excel documentation)
                        "position": chart_object.Left,  # Left position of the chart object
                        "size": (chart_object.Width, chart_object.Height),  # Width and height of the chart object
                        "has_legend": chart.HasLegend,
                        "data_source": "",
                        "axes":[],
                        "legend": legend_info if len(legend_info) > 0 else None# Boolean indicating presence of a legend
                    } 
                    # print(workbook_info)
                    try:
                    # Attempt using chart.ChartArea.LinkedSourceName (may not be available in all versions)
                        workbook_info[sheet.Name]["charts"][chart.ChartTitle.Text][-1]["data_source"] = chart.ChartArea.LinkedSourceName
                    except:
                        pass 

                    try:
                    # Attempt using chart.ChartArea.ChartGroups(1).SourceName (may not be available in all versions)
                        workbook_info[sheet.Name]["charts"][chart.ChartTitle.Text][-1]["data_source"] = chart.ChartArea.ChartGroups(1).SourceName  # Assuming first chart group
                    except:
                        pass

                    # Extract axis information (assuming two axes - X and Y)
                    for axis_id in range(0, 2):
                        axis = chart.Axes(axis_id + 1)
                        if axis.HasTitle:
                            axis_info = {
                                "axis_type": axis.AxisType,
                                "title": axis.AxisTitle.Text,
                            }
                            workbook_info[sheet.Name]["charts"][chart.ChartTitle.Text]["axes"].append(axis_info)
                workbook_info[sheet.Name]["pivot_tables"] = {}
                for pivot_table in sheet.PivotTables():
                    workbook_info[sheet.Name]["pivot_tables"][pivot_table.Name] = {
                        "name": pivot_table.Name,
                        "data_source": pivot_table.CacheSource if hasattr(pivot_table, 'CacheSource') else None,  # Data source name
                        "row_fields": [],  # Placeholder for row fields
                        "column_fields": [],  # Placeholder for column fields
                        "value_fields": [],  # Placeholder for value fields
                        "filters": [],  # Placeholder for filters (optional)
                        "summary_types": [],  # Placeholder for summary types
                    }
                    # Extract row fields
                    for field in pivot_table.PivotFields():
                        # print(pivot_table.PivotFields(str(field)).Orientation)
                        if pivot_table.PivotFields(str(field)).Orientation == win32c.PivotFieldOrientation.xlRowField:  # Check for row field type
                            workbook_info[sheet.Name]["pivot_tables"][pivot_table.Name]["row_fields"].append(pivot_table.PivotFields(str(field)).Name)
                        elif pivot_table.PivotFields(str(field)).Orientation == win32c.PivotFieldOrientation.xlColumnField:  # Check for column field type
                            workbook_info[sheet.Name]["pivot_tables"][pivot_table.Name]["column_fields"].append(pivot_table.PivotFields(str(field)).Name)
                        else:  # Check for value field type
                            workbook_info[sheet.Name]["pivot_tables"][pivot_table.Name]["value_fields"].append(pivot_table.PivotFields(str(field)).Name)
                            # workbook_info[sheet.Name]["pivot_tables"][pivot_table.Name]["summary_types"].append(str(pivot_table.PivotFields(str(field)).Function))  # Summary type
                workbook_info[sheet.Name]["formulas"] = {}
                for cell in sheet.UsedRange:
                    if cell.HasFormula:  # Check if cell contains a formula
                        workbook_info[sheet.Name]["formulas"][cell.Formula] = {
                        "cell": cell.Address,
                        "formula": cell.Formula,  # Formula string
                        "source_ranges": [],  # Placeholder for referenced ranges
                        "destination_range": cell.Address,  # Destination (cell itself)
                        }

                        # Extract referenced ranges (may require parsing the formula string)
                        for argument_index in range(cell.FormulaR1C1.Arguments.Count):
                            argument = cell.FormulaR1C1.Arguments(argument_index)
                        if argument.IsRange:  # Check if argument is a range reference
                            workbook_info[sheet.Name]["formulas"][cell.Formula]["source_ranges"].append(argument.Text)
                workbook_info[sheet.Name]["merged_cells"] = []      
                for row in range(1, sheet.UsedRange.Rows.Count + 1):
                    for col in range(1, sheet.UsedRange.Columns.Count + 1):
                        cell = sheet.Cells(row, col)
                        if cell.MergeCells:
                            workbook_info[sheet.Name]["merged_cells"].append((cell.Address, cell.Value))
            wb.Close(SaveChanges = True)
                
                
                
    def _destruct(self):
        self.app.Quit()    