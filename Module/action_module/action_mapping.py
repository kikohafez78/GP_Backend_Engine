from actions.charts import Charts_App
from actions.entryandmanipulation import entry_manipulation_App
from actions.formatting import formatting_App
from actions.management import management_App
from actions.pivot_table import Pivot_App
from actions.formula import Formula_App
from typing import List, Optional, Any
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

def class_mapping():
    return {
        "entry and manipulation" : entry_manipulation_App(),
        "management" : management_App(),
        "formatting": formatting_App(),
        "charts" : Charts_App(),
        "pivot tables": Pivot_App(),
        "formula" : Formula_App()
    }
    
def action_mapping(class_: Any):
    if isinstance(class_ ,entry_manipulation_App):
        class_: entry_manipulation_App = class_
        return {
            "update cell range" : class_.create_sheet,
            "delete cells": class_.delete_cells,
            "merge cells": class_.merge_cells,
            "unmerge cells": class_.unmerge_cells,
            "split text to columns": class_.split_text_to_columns,
            "insert rows": class_.insert_rows,
            "insert columns": class_.insert_columns,
            "autofill": class_.autofill,
            "copy-paste": class_.copy_paste,
            "copy-paste format": class_.copy_paste_format,
            "copy sheet": class_.copy_sheet,
            "cut-paste": class_.cut_paste,
            "find and replace": class_.find_and_replace,
            "set hyperlink": class_.set_hyperlink,
            "delete hyperlink": class_.delete_hyperlink,
            "remove duplicates":class_.remove_duplicates,
            "rename sheet": class_.insert_checkbox,
            "create sheet": class_.create_sheet,
            "delete sheet": class_.delete_sheet,
            "clear": class_.clear
        }
    elif isinstance(class_ , management_App):
        class_: management_App = class_
        return {
            "switch sheet": class_.switch_sheet,
            "sort": class_.sort,
            "filter": class_.filter,
            "delete filter": class_.delete_filter,
            "slicer": class_.slicer,
            "move rows": class_.move_rows,
            'move columns': class_.move_columns,
            'group': class_.group,
            'ungroup': class_.ungroup,
            'hide rows': class_.hide_unhide_rows,
            'unhide rows': class_.hide_unhide_rows,
            'hide columns': class_.hide_unhide_columns,
            'unhide columns': class_.hide_unhide_columns,
            'hide sheet': class_.hide_unhide_sheet,
            'unhide sheet': class_.hide_unhide_sheet,
            'set password': class_.set_password,
            'transpose': class_.transpose,
            'create named range': class_.create_named_range,
            'freeze panes': class_.freeze_panes,
            'unfreeze panes': class_.unfreeze_panes,
            'split panes': class_.split_panes,
            'data consolidation': class_.data_consolidation
        }
    elif isinstance(class_, formatting_App):
        class_: formatting_App = class_
        return {
            'format cells': class_.format_cells,
            'delete format': class_.delete_format,
            'set data type': class_.set_data_type,
            'change page layout': class_.change_page_layout,
            'set border': class_.set_border,
            'data validation': class_.data_validation,
            'display formula': class_.display_formula,
            'wrap text': class_.wrap_unwrap_text,
            "unwrap text": class_.wrap_unwrap_text,
            'autofit': class_.autofit,
            'resize cells': class_.resize_cells,
            'conditional formatting': class_.conditional_formatting,
            'lock unlock cells': class_.lock_unlock_cells,
            'unlock cells': class_.lock_unlock_cells,
            'protect cells': class_.protect_unprotect_cells,
            "unprotect cells": class_.protect_unprotect_cells,
            'dropdown list': class_.dropdown_list
        }
    elif isinstance(class_, Charts_App):
        class_: Charts_App = class_
        return {
                "create chart": class_.CreateChart,
                "set chart trendline": class_.SetChartTrendline,
                "set chart title": class_.SetChartTitle,
                "set chart has axis": class_.SetChartHasAxis,
                "set chart axis": class_.SetChartAxis,
                "set chart legend": class_.SetChartLegend,
                "set chart has legend": class_.SetChartHasLegend,
                "set chart type": class_.SetChartType,
                "set chart source": class_.SetChartSource,
                "set chart background color": class_.SetChartBackgroundColor,
                "resize chart": class_.ResizeChart,
                "set chart data color": class_.SetChartDataColor,
                "highlight data points": class_.HighlightDataPoints,
                "set data series type": class_.SetDataSeriesType,
                "add data series": class_.AddDataSeries,
                "remove data series": class_.RemoveDataSeries,
                "set data series source": class_.SetDataSeriesSource,
                "add chart error bars": class_.AddChartErrorBars,
                "add chart error bar": class_.AddChartErrorBar,
                "remove chart error bars": class_.RemoveChartErrorBars,
                "remove chart error bar": class_.RemoveChartErrorBar,
                "add data labels": class_.AddDataLabels,
                "remove data labels": class_.RemoveDataLabels,
                "set chart marker": class_.SetChartMarker,
                "copy paste chart": class_.CopyPasteChart
            }

    elif isinstance(class_, Pivot_App):
        class_: Pivot_App = class_
        return {
            "create pivot table": class_.CreatePivotTable,
            "set summary type": class_.set_summary_type,
            "sort pivot table": class_.sort_pivot_table,
            "remove pivot table": class_.remove_pivot_table
        }
        
        
def create_function_dict(functions_code):
    import re

    func_pattern = re.compile(r'def (\w+)\(')
    functions = func_pattern.findall(functions_code)

    function_dict = {}
    for func in functions:
        formatted_key = ' '.join(word.capitalize() for word in func.split('_'))
        function_dict[formatted_key] = f'class_.{func}'

    return function_dict



def get_chart_arguments_mapping():
    return {
            'CreateChart': {
                'src_wb_path': str,
                'source': str,
                'destSheet': str,
                'chartType': str,
                'chartName': str,
                'XField': Optional[int],
                'YField': List[int],
                'output_wb_path': Optional[str]
            },
            'SetChartTrendline': {
                'src_wb_path': str,
                'chartName': str,
                'trendlineType': List[str],
                'DisplayEquation': Optional[bool],
                'DisplayRSquared': Optional[bool],
                'output_wb_path': Optional[str]
            },
            'SetChartTitle': {
                'src_wb_path': str,
                'chartName': str,
                'title': str,
                'fontSize': Optional[float],
                'bold': Optional[bool],
                'color': Optional[int],
                'output_wb_path': Optional[str]
            },
            'SetChartHasAxis': {
                'src_wb_path': str,
                'chartName': str,
                'axis': str,
                'hasAxis': bool,
                'output_wb_path': Optional[str]
            },
            'SetChartAxis': {
                'src_wb_path': str,
                'chartName': str,
                'axis': str,
                'title': Optional[str],
                'labelOrientation': Optional[str],
                'maxValue': Optional[float],
                'miniValue': Optional[float],
                'output_wb_path': Optional[str]
            },
            'SetChartLegend': {
                'src_wb_path': str,
                'chartName': str,
                'position': Optional[str],
                'fontSize': Optional[str],
                'seriesName': Optional[list],
                'output_wb_path': Optional[str]
            },
            'SetChartHasLegend': {
                'src_wb_path': str,
                'chartName': str,
                'hasLegend': bool,
                'output_wb_path': Optional[str]
            },
            'SetChartType': {
                'src_wb_path': str,
                'chartName': str,
                'chartType': str,
                'output_wb_path': Optional[str]
            },
            'SetChartSource': {
                'chartName': str,
                'source': str
            },
            'SetChartBackgroundColor': {
                'src_wb_path': str,
                'chartName': str,
                'color': str,
                'output_wb_path': Optional[str]
            },
            'ResizeChart': {
                'src_wb_path': str,
                'chartName': str,
                'width': float,
                'height': float,
                'output_wb_path': Optional[str]
            },
            'SetChartDataColor': {
                'src_wb_path': str,
                'chartName': str,
                'colorRGB': list,
                'output_wb_path': Optional[str]
            },
            'HighlightDataPoints': {
                'src_wb_path': str,
                'chartName': str,
                'pointIndex': int,
                'colorRGB': list,
                'output_wb_path': Optional[str]
            },
            'SetDataSeriesType': {
                'src_wb_path': str,
                'chartName': str,
                'seriesIndex': int,
                'seriesType': str,
                'output_wb_path': Optional[str]
            },
            'AddDataSeries': {
                'src_wb_path': str,
                'src_sheet': str,
                'chartName': str,
                'xrange': str,
                'yrange': str,
                'output_wb_path': Optional[str]
            },
            'RemoveDataSeries': {
                'src_wb_path': str,
                'chartName': str,
                'seriesIndex': int,
                'output_wb_path': Optional[str]
            },
            'SetDataSeriesSource': {
                'src_wb_path': str,
                'chartName': str,
                'seriesIndex': int,
                'xrange': str,
                'yrange': str,
                'output_wb_path': Optional[str]
            },
            'AddChartErrorBars': {
                'src_wb_path': str,
                'chartName': str,
                'output_wb_path': Optional[str]
            },
            'AddChartErrorBar': {
                'src_wb_path': str,
                'chartName': str,
                'seriesIndex': int,
                'output_wb_path': Optional[str]
            },
            'RemoveChartErrorBars': {
                'src_wb_path': str,
                'chartName': str,
                'output_wb_path': Optional[str]
            },
            'RemoveChartErrorBar': {
                'src_wb_path': str,
                'chartName': str,
                'seriesIndex': int,
                'output_wb_path': Optional[str]
            },
            'AddDataLabels': {
                'src_wb_path': str,
                'chartName': str,
                'output_wb_path': Optional[str]
            },
            'RemoveDataLabels': {
                'src_wb_path': str,
                'chartName': str,
                'output_wb_path': Optional[str]
            },
            'SetChartMarker': {
                'src_wb_path': str,
                'chartName': str,
                'style': Optional[List[str]],
                'size': Optional[float],
                'output_wb_path': Optional[str]
            },
            'CopyPasteChart': {
                'src_wb_path': str,
                'chartName': str,
                'dest_sheet': str,
                'destination': str,
                'output_wb_path': Optional[str]
            }
        }

def get_pivot_table_arguments_mapping():
    return  {
            "CreatePivotTable": {
                "src_wb_path": str,
                "src_wb_sheet": str,
                "source": str,
                "destSheet": str,
                "name": str,
                "RowField": List,
                "ColumnField": List,
                "PageField": List,
                "DataField": List,
                "summarizeFunction": str,
                "output_wb_path": str or None
            },
            "GetBlankArea": {
                "sheetName": str
            },
            "remove_pivot_table": {
                "src_wb_path": str,
                "name": str,
                "output_wb_path": str or None
            },
            "set_summary_type": {
                "src_wb_path": str,
                "name": str,
                "field": str,
                "func": str,
                "output_wb_path": str or None
            },
            "sort_pivot_table": {
                "src_wb_path": str,
                "name": str,
                "field": str,
                "key": str,
                "order": str,
                "output_wb_path": str or None
            }
        }
    
def get_entry_arguments_mapping():
    return  {
            "update_cell_value": {
                "workbook_path": str,
                "destination_sheet": str,
                "cell_range": str,
                "value": None,
                "output_workbook_path": str
            },
            "delete_cells": {
                "workbook_path": str,
                "sheet_name": str,
                "cell_range": str,
                "output_workbook_path": str
            },
            "merge_cells": {
                "workbook_path": str,
                "sheet_name": str,
                "cell_range": str,
                "output_workbook_path": str
            },
            "unmerge_cells": {
                "workbook_path": str,
                "sheet_name": str,
                "cell_range": str,
                "output_workbook_path": str
            },
            "split_text_to_columns": {
                "workbook_path": str,
                "sheet_name": str,
                "cell_range": str,
                "output_workbook_path": str,
                "delimiter": str
            },
            "insert_rows": {
                "workbook_path": str,
                "sheet_name": str,
                "above_row": int,
                "below_row": int,
                "count": int,
                "output_workbook_path": str
            },
            "insert_columns": {
                "workbook_path": str,
                "sheet_name": str,
                "before_column": int,
                "after_column": int,
                "count": int,
                "output_workbook_path": str
            },
            "autofill": {
                "workbook_path": str,
                "sheet_name": str,
                "cell_range": str,
                "output_workbook_path": str
            },
            "copy_paste": {
                "source_workbook_path": str,
                "source_sheet_name": str,
                "target_sheet_name": str,
                "source_range": str,
                "target_range": str,
                "output_workbook_path": str
            },
            "copy_paste_format": {
                "source_workbook_path": str,
                "source_sheet_name": str,
                "target_sheet_name": str,
                "source_range": str,
                "target_range": str,
                "output_workbook_path": str
            },
            "copy_sheet": {
                "source_workbook_file": str,
                "sheet_name": str,
                "new_sheet_name": str,
                "output_workbook_path": str
            },
            "cut_paste": {
                "src_workbook_path": str,
                "tgt_workbook_path": str,
                "src_sheet_name": str,
                "tgt_sheet_name": str,
                "source_range": str,
                "target_range": str
            },
            "find_n_replace": {
                "workbook_path": str,
                "sheet_name": str,
                "find_text": str,
                "replace_text": str,
                "output_worbook_path": str
            },
            "set_hyperlink": {
                "workbook_path": str,
                "sheet_name": str,
                "cell_range": str,
                "url": str,
                "output_workbook_path": str
            },
            "delete_hyperlink": {
                "workbook_path": str,
                "sheet_name": str,
                "cell_range": str,
                "output_workbook_path": str
            },
            "remove_duplicates": {
                "workbook_path": str,
                "sheet_name": str,
                "column_number": int,
                "cell_range": str,
                "target_workbook_path": str
            },
            "rename_sheet": {
                "workbook_path": str,
                "old_sheet_name": str,
                "new_sheet_name": str,
                "target_workbook_path": str
            },
            "insert_checkbox": {
                "workbook_path": str,
                "sheet_name": str,
                "cell_range": str,
                "output_workbook_path": str
            },
            "insert_textbox": {
                "workbook_path": str,
                "sheet_name": str,
                "cell_range": str,
                "output_workbook_path": str,
                "text": str
            },
            "create_sheet": {
                "workbook_path": str,
                "sheet_name": str
            },
            "delete_sheet": {
                "workbook_path": str,
                "sheet_name": str
            },
            "clear": {
                "workbook_path": str,
                "sheet_name": str,
                "cell_range": str,
                "output_workbook_path": str
            }
        }

def get_management_arguments_mapping():
    return {
            "sort": {
                "workbook_path": str,
                "sheet_name": str,
                "cell_range": str,
                "order": str,
                "orientation": str,
                "output_workbook_path": str
            },
            "filter": {
                "workbook_path": str,
                "sheet_name": str,
                "cell_range": str,
                "feild_index": int,
                "criteria": None,
                "output_workbook_path": str
            },
            "delete_filter": {
                "workbook_path": str,
                "sheet_name": str,
                "output_workbook_path": str
            },
            "slicer": {
                "workbook_path": str,
                "source_sheet_name": str,
                "source_range": str,
                "key": str,
                "value": str,
                "output_workbook_path": str
            },
            "move_rows": {
                "workbook_path": str,
                "source_sheet_name": str,
                "row_range": tuple,
                "new_pos": int,
                "output_workbook_path": str
            },
            "move_columns": {
                "workbook_path": str,
                "source_sheet_name": str,
                "column_range": tuple,
                "new_pos": int,
                "output_workbook_path": str
            },
            "group": {
                "workbook_path": str,
                "source_sheet_name": str,
                "source_range": str,
                "output_workbook_path": str,
                "group_by_rows": bool,
                "hidden": bool
            },
            "ungroup": {
                "workbook_path": str,
                "source_sheet_name": str,
                "source_range": str,
                "output_workbook_path": str,
                "group_by_rows": bool
            },
            "hide_unhide_rows": {
                "workbook_path": str,
                "source_sheet_name": str,
                "start_row": None,
                "end_row": None,
                "output_workbook_path": str,
                "hidden": bool
            },
            "hide_unhide_columns": {
                "workbook_path": str,
                "source_sheet_name": str,
                "start_col": None,
                "end_col": None,
                "output_workbook_path": str,
                "hidden": bool
            },
            "hide_unhide_sheet": {
                "workbook_path": str,
                "source_sheet_name": str,
                "hidden": bool,
                "output_workbook_path": str
            },
            "set_password": {
                "workbook_path": str,
                "source_sheet_name": str,
                "password": str,
                "output_workbook_path": str,
                "wb": bool
            },
            "transpose": {
                "workbook_path": str,
                "source_sheet_name": str,
                "source_range": str,
                "output_workbook_path": str
            },
            "create_named_range": {
                "workbook_path": str,
                "source_sheet_name": str,
                "range_name": str,
                "source_range": str,
                "output_workbook_path": str
            },
            "freeze_panes": {
                "workbook_path": str,
                "sheet_name": str,
                "range_obj": str,
                "output_workbook_path": str
            },
            "unfreeze_panes": {
                "workbook_path": str,
                "sheet_name": str,
                "output_workbook_path": str
            },
            "split_panes": {
                "workbook_path": str,
                "sheet_name": str,
                "vertical_split": str,
                "horizontal_split": str,
                "output_workbook_path": str
            },
            "data_consolidation": {
                "workbook_path": str,
                "ranges": None,
                "destination_sheet": str,
                "output_workbook_path": str
            }
        }
    
def get_formatting_arguments_mapping():
    return  {
            "format_cells": {
                "workbook_path": str,
                "sheet_name": str,
                "cell_range": str,
                "font": Optional[str],
                "fontSize": Optional[float],
                "color": Optional[int],
                "fillColor": Optional[int],
                "bold": Optional[bool],
                "italic": Optional[bool],
                "underline": Optional[bool],
                "horizontalAlignment": Optional[str],
                "output_workbook_path": str
            },
            "delete_format": {
                "workbook_path": str,
                "sheet_name": str,
                "cell_range": str,
                "output_workbook_path": str
            },
            "set_data_type": {
                "workbook_path": str,
                "sheet_name": str,
                "dataType": str,
                "cell_range": str,
                "output_workbook_path": str
            },
            "change_page_layout": {
                "workbook_path": str,
                "sheet_name": str,
                "paper_size": str,
                "orientation": str,
                "output_workbook_path": str
            },
            "set_border": {
                "workbook_path": str,
                "sheet_name": str,
                "cell_range": str,
                "color": str,
                "weight": str,
                "output_workbook_path": Optional[str]
            },
            "data_validation": {
                "workbook_path": str,
                "sheet_name": str,
                "cell_range": str,
                "type": str,
                "formula1": str,
                "output_workbook_path": Optional[str]
            },
            "display_formula": {
                "workbook_path": str,
                "display": bool,
                "output_workbook_path": Optional[str]
            },
            "wrap_unwrap_text": {
                "workbook_path": str,
                "sheet_name": str,
                "cell_range": str,
                "wrap": bool,
                "output_workbook_path": Optional[str]
            },
            "autofit": {
                "workbook_path": str,
                "sheet_name": str,
                "cell_range": str,
                "output_workbook_path": Optional[str]
            },
            "resize_cells": {
                "workbook_path": str,
                "sheet_name": str,
                "cell_range": str,
                "width": Optional[int],
                "height": Optional[int],
                "output_workbook_path": Optional[str]
            },
            "conditional_formatting": {
                "workbook_path": str,
                "sheet_name": str,
                "cell_range": str,
                "formula": str,
                "bold": Optional[bool],
                "color": Optional[str],
                "fillColor": Optional[str],
                "italic": Optional[bool],
                "underline": Optional[bool],
                "output_workbook_path": Optional[str]
            },
            "lock_unlock_cells": {
                "workbook_path": str,
                "sheet_name": str,
                "cell_range": str,
                "lock": bool,
                "output_workbook_path": Optional[str]
            },
            "protect_unprotect_cells": {
                "workbook_path": str,
                "sheet_name": str,
                "cell_range": str,
                "protect": bool,
                "password": str,
                "output_workbook_path": Optional[str]
            },
            "dropdown_list": {
                "workbook_path": str,
                "sheet_name": str,
                "cell_range": str,
                "dropdown_values": List[str],
                "output_workbook_path": Optional[str]
            }
        }
