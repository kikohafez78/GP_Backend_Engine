import pandas as pd
import openai

key = "sk-proj-D5NOrqAu8vEtOEHO4PlTT3BlbkFJD5PszbRHSIzKs69KudSg"
actions = """
TODAY: Returns the current date.
NOW: Returns the current date and time.
DATE: Creates a date value given a year, month, and day.
TIME: Creates a time value given an hour, minute, and second.
YEAR: Extracts the year from a given date.
MONTH: Extracts the month from a given date.
DAY: Extracts the day of the month from a given date.
HOUR: Extracts the hour from a given time.
MINUTE: Extracts the minute from a given time.
SECOND: Extracts the second from a given time.
DATEDIF: Calculates the difference between two dates in days, months, or years.
DATEDIF: Calculates the number of days between two dates.
DATEDIF: Calculates the number of complete months between two dates.
DATEDIF: Calculates the number of complete years between two dates.
DATEDIF: Calculates the difference between two dates in a specified time unit (e.g., "m" for months, "y" for years).
DATEVALUE: Converts a date string into a serial number that Excel recognizes as a date.
TIMEVALUE: Converts a time string into a serial number that Excel recognizes as a time.
EOMONTH: Returns the last day of the month, a specified number of months before or after a given date.
WORKDAY: Calculates the date of the Nth workday before or after a given date, excluding weekends and specified holidays.
NETWORKDAYS: Calculates the number of workdays between two dates, excluding weekends and specified holidays.
IF: Evaluates a specified condition and returns one value if the condition is true and another value if the condition is false for a range of cells.
AND: Returns TRUE if all the supplied arguments evaluate to TRUE; otherwise, it returns FALSE for a range of cells.
OR: Returns TRUE if any of the supplied arguments evaluate to TRUE; otherwise, it returns FALSE  for a range of cells.
NOT: Returns the opposite of a logical value; it converts TRUE to FALSE and FALSE to TRUE for a range of cells.
IFERROR: Checks whether a formula results in an error and returns a specified value if an error is encountered; otherwise, it returns the result of the formula for a range of cells.
IFNA: Checks whether a formula results in the #N/A error value and returns a specified value if #N/A is encountered; otherwise, it returns the result of the formula for a range of cells.
IFS: Checks multiple conditions and returns a value that corresponds to the first TRUE condition; it replaces nested IF statements for a range of cells.
SWITCH: Evaluates an expression against a list of values and returns a corresponding result; it replaces multiple IF statements for a range of cells.
XOR: Returns TRUE if an odd number of arguments evaluate to TRUE; otherwise, it returns FALSE for a range of cells.
ISEVEN: Checks whether a number is even and returns TRUE if the number is even; otherwise, it returns FALSE for a range of cells.
ISODD: Checks whether a number is odd and returns TRUE if the number is odd; otherwise, it returns FALSE for a range of cells.
IFERROR: Checks whether a formula results in an error and returns a specified value if an error is encountered; otherwise, it returns the result of the formula for a range of cells.
IFNA: Checks whether a formula results in the #N/A error value and returns a specified value if #N/A is encountered; otherwise, it returns the result of the formula for a range of cells.
ISEMPTY: Checks whether a specified cell is empty and returns TRUE if the cell is empty; otherwise, it returns FALSE for a range of cells.
ISBLANK: Checks whether a specified cell is blank (i.e., contains no value) and returns TRUE if the cell is blank; otherwise, it returns FALSE for a range of cells.
ADDITION: add a number of cells to another number of cells or to a scalar and store the result in a range of target cells
SUBTRACTION: subtract a number of cells from another number of cells or from a scalar and store the result in a range of target cells
MULTIPLICATION: multiply a number of cells by another number of cells or by a scalar and store the result in a range of target cells
DIVISION: divide a number of cells by another number of cells or by a scalar and store the result in a range of target cells
SUM: Calculates the sum of a range of cells.
AVERAGE: Calculates the average of a range of cells.
MIN: Returns the minimum value from a range of cells.
MAX: Returns the maximum value from a range of cells.
COUNT: Counts the number of cells containing numbers in a range for a range of cells.
COUNTA: Counts the number of non-empty cells in a range for a range of cells.
COUNTIF: Counts the number of cells in a range that meet a specific condition for a range of cells.
SUMIF: Adds the cells in a range that meet a specific condition for a range of cells.
AVERAGEIF: Calculates the average of cells in a range that meet a specific condition for a range of cells.
SUMIFS: Adds the cells in a range that meet multiple conditions for a range of cells.
AVERAGEIFS: Calculates the average of cells in a range that meet multiple conditions for a range of cells.
ABS: Returns the absolute value of a number for a range of cells.
ROUND: Rounds a number to a specified number of digits for a range of cells.
ROUNDUP: Rounds a number up, away from zero, to a specified number of digits for a range of cells.
ROUNDDOWN: Rounds a number down, towards zero, to a specified number of digits for a range of cells.
INT: Rounds a number down to the nearest integer for a range of cells.
MOD: Returns the remainder of a division operation for a range of cells.
SQRT: Returns the square root of a number for a range of cells.
POWER: Raises a number to a specified power for a range of cells.
LOG: Returns the logarithm of a number to a specified base for a range of cells.
EXP: Returns e raised to the power of a specified number for a range of cells.
RANDBETWEEN: Returns a random integer between two specified numbers.
TRUNC: Truncates a number to a specified number of digits for a range of cells.
LN: Returns the natural logarithm of a number for a range of cells.
LOG10: Returns the base-10 logarithm of a number for a range of cells.
CEILING: Rounds a number up, towards positive infinity, to the nearest multiple of significance for a range of cells.
FLOOR: Rounds a number down, towards negative infinity, to the nearest multiple of significance for a range of cells
SIGN: Returns the sign of a number (1 for positive, -1 for negative, 0 for zero) for a range of cells.
MEDIAN: Calculates the median (middle value) of a range of values from a range of cells.
MODE: Calculates the mode (most frequently occurring value) of a range of values from a range of cells.
MIN: Returns the minimum value from a range of cells.
MAX: Returns the maximum value from a range of cells.
STDEV.S: Calculates the standard deviation based on a sample of data from a range of cells.
STDEV.P: Calculates the standard deviation of an entire population of data from a range of cells.
VAR.S: Calculates the variance based on a sample of data from a range of cells.
VAR.P: Calculates the variance of an entire population of data from a range of cells.
COUNT: Counts the number of cells containing numerical values in a range of cells.
COUNTA: Counts the number of non-empty cells in a range.
COUNTIF: Counts the number of cells in a range that meet a specific condition.
SUM: Calculates the sum of a range of cells.
SUMIF: Adds the cells in a range of cells that meet a specific condition.
AVERAGEIF: Calculates the average in a range of cells if that cell meet a specific condition.
CORREL: Calculates the correlation coefficient between two sets of values.
COVARIANCE.S: Calculates the covariance between two ranges of cells based on a sample range of cells.
COVARIANCE.P: Calculates the covariance between two ranges of cells for an entire population.
PERCENTILE: Returns the value at a specified percentile in a range of cells.
QUARTILE: Returns the quartile (25th, 50th, or 75th percentile) of a range of cells.
RANK: Returns the rank of a value in a range of cells.
FORECAST: Calculates a future value based on existing values using linear regression for a range of cells.
GROWTH: Calculates predicted exponential growth based on existing values.
TREND: Calculates linear trend values based on existing values in a range of cells.
CHISQ.DIST: Returns the chi-square distribution probability for a range of cells.
Z.TEST: Returns the one-tailed probability value of a z-test for a range of cells.
T.TEST: Returns the probability associated with a t-test for a range of cells.
F.TEST: Returns the result of an F-test for a range of cells.
NORM.DIST: Returns the normal distribution probability for given values  from a range of cells.
NORM.INV: Returns the inverse of the normal distribution function for a given probability .
CONCATENATE: Combines two or more text strings from different cells into one.
LEN: Returns the length of a text string from a range of cells.
LEFT: Returns the leftmost characters from a text string from a range of cells.
RIGHT: Returns the rightmost characters from a text string from a range of cells.
MID: Returns a specific number of characters from the middle of a text string from a range of cells.
LOWER: Converts text to lowercase for a range of cells.
UPPER: Converts text to uppercase for a range of cells.
PROPER: Capitalizes the first letter of each word in a text string in a range of cells.
TRIM: Removes leading and trailing spaces from a text string in a range of cells.
SUBSTITUTE: Replaces occurrences of a specified substring with another substring in a text string in a range of cells.
FIND: Returns the starting position of one text string within another text string in a range of cells.
SEARCH: Returns the starting position of one text string within another text string (case-insensitive) in range of cells.
REPLACE: Replaces part of a text string with another text string, starting at a specified position in a range of cells.
REPT: Repeats a text string a specified number of times in a range of cells.
TEXT: Formats a number as text using a specified format in a range of cells.
VALUE: Converts a text string that represents a number to a number in a range of cells.
CONCAT: Concatenates a list of text strings from a range of cells.
EXACT: Compares two ranges of cells and returns TRUE if they are exactly the same, otherwise returns FALSE.
LEFTB: Returns the leftmost characters from a text string, based on the number of bytes from a range of cells.
RIGHTB: Returns the rightmost characters from a text string, based on the number of bytes from a range of cells.
MIDB: Returns a specific number of characters from the middle of a text string, based on the number of bytes from a range of cells.
LENB: Returns the length of a text string, in bytes from a range of cells.
FIND: Returns the starting position of one text string within another text string in a range of cells.
SEARCH: Returns the starting position of one text string within another text string (case-insensitive) in a range of cells.
REPLACE: Replaces part of a text string with another text string, starting at a specified position.
REPLACEB: Replaces part of a text string with another text string, based on the number of bytes in a range of cells.
TEXTJOIN: Joins multiple text strings into one text string, with a specified delimiter from a range of cells.
SUBSTITUTE: Replaces occurrences of a specified substring with another substring in a text string, based on the number of occurrences to replac in a range of cells.
CHAR: Returns the character specified by a numeric code  from a range of cells.
CLEAN: Removes non-printable characters from a text string  from a range of cells.

"""
start_row = 6801  # Adjust this if you have existing data in the DataFrame
openai.api_key = key
# Option 1: Pre-allocate rows (more efficient)
# df = df.reindex(range(start_row, start_row + 100 * len(chart_functions)))
prefix = """
imagine you are an excel user and you are chatting with an ai model that takes in nl prompts and executes the intention behind it, I want you to generate a 100 different prompts containing the sole intention and assume any arguments for the intention including chart type, data source,data axis,  etc.. :
""" 
df = pd.DataFrame(columns=["prompt", "intent"])
actions = actions.splitlines()
i = 0
while i < len(actions):
    # Create a temporary DataFrame with 100 rows and 1 column
    message = prefix + "\n" + actions[i] 
    intent = actions[i].split(":")[0]
    response = openai.chat.completions.create(
              model="gpt-3.5-turbo",
              messages=[{"role": "user", "content": message}])
    temp_df = pd.DataFrame(data = response.choices[0].message.content.splitlines(), columns=["prompt"])
    if len(temp_df) == 100:  
    # Option 2: Iterate and assign individual rows (safer for empty slices)
        for j in range(100):
            df.loc[start_row + (i * 100) + j, 'prompt'] = temp_df.loc[j, 'prompt']
        i += 1
    print(i, len(temp_df))
df.to_csv("./Book2.csv")