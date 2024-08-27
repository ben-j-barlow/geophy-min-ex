import re

# Function to process the log file and extract unique function calls
def extract_unique_functions(log_file, output_file):
    # Read the log file
    with open(log_file, 'r') as f:
        log_lines = f.readlines()
    
    # Initialize a set to store unique function call pairs
    unique_functions = set()

    # Regex to match function call lines
    function_regex = re.compile(r"┌ Info: (.+)$")

    # Extract function calls
    i = 0
    while i < len(log_lines) - 1:
        if function_regex.search(log_lines[i]):
            function_call = log_lines[i].strip() + "\n" + log_lines[i + 1].strip()
            unique_functions.add(function_call)
            i += 2  # Move to the next pair
        else:
            i += 1  # Move to the next line

    # Write unique function calls to the output file
    with open(output_file, 'w') as f:
        for func in unique_functions:
            f.write(f"{func}\n\n")  # Add an extra newline for separation


def extract_functions_in_order(log_file, output_file):
    # Read the log file
    with open(log_file, 'r') as f:
        log_lines = f.readlines()
    
    # Initialize a list to store function call pairs
    function_calls = []

    # Regex to match function call lines
    function_regex = re.compile(r"┌ Info: (.+)$")

    # Extract function calls
    i = 0
    last_function_call = None
    while i < len(log_lines) - 1:
        if function_regex.search(log_lines[i]):
            function_call = log_lines[i].strip() + "\n" + log_lines[i + 1].strip()
            if function_call != last_function_call:
                function_calls.append(function_call)
                last_function_call = function_call
            i += 2  # Move to the next pair
        else:
            i += 1  # Move to the next line

    # Write function calls to the output file
    with open(output_file, 'w') as f:
        for func in function_calls:
            f.write(f"{func}\n\n")  # Add an extra newline for separation



# Specify the log file path
#log_file_path = "logging/run_pomcpow_trial_mfms.log"  # Replace with your actual log file path

#output_file_path = "logging/set_run_pomcpow_trial_mfms.log"  # Replace with your actual log file path

# Extract unique functions and write to the output file

#extract_unique_functions(log_file_path, output_file_path)
#extract_functions_in_order(log_file_path, output_file_path)