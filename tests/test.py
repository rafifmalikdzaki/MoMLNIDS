# Reading the newly uploaded log file
import pandas as pd
file_path_new = '../manualisasi.log'
with open(file_path_new, 'r') as file:
    new_log_content = file.readlines()

# Re-initialize the data storage structure
parsed_layers_data = []
current_layer = {}
capture_value = False
value_type = ""

# Parsing the new log content for structured extraction
for line in new_log_content:
    stripped_line = line.strip()
    
    # Identify new layer start
    if stripped_line.startswith("Layer:"):
        if current_layer:
            # Append previous layer info to parsed_layers_data before starting a new one
            parsed_layers_data.append(current_layer)
        # Start capturing new layer data
        current_layer = {
            "Layer Name": stripped_line.split(": ")[1].split(" (")[0],
            "Layer Type": stripped_line.split("(")[-1].strip(")"),
            "Output": [],
            "Weight": [],
            "Bias": [],
            "Weight Gradient": [],
            "Bias Gradient": []
        }
        capture_value = False
    
    # Capture output, weights, bias, or gradients
    elif "Output:" in stripped_line:
        capture_value = True
        value_type = "Output"
        current_layer["Output"] = [stripped_line.split("Output: ")[1]]
    elif "Weight:" in stripped_line:
        capture_value = True
        value_type = "Weight"
        current_layer["Weight"] = [stripped_line.split("Weight: ")[1]]
    elif "Bias:" in stripped_line and "Gradient" not in stripped_line:
        capture_value = True
        value_type = "Bias"
        current_layer["Bias"] = [stripped_line.split("Bias: ")[1]]
    elif "Weight Gradient:" in stripped_line:
        capture_value = True
        value_type = "Weight Gradient"
        current_layer["Weight Gradient"] = [stripped_line.split("Weight Gradient: ")[1]]
    elif "Bias Gradient:" in stripped_line:
        capture_value = True
        value_type = "Bias Gradient"
        current_layer["Bias Gradient"] = [stripped_line.split("Bias Gradient: ")[1]]
    
    # Continue capturing multi-line values (for tensor values across lines)
    elif capture_value:
        if value_type and stripped_line:  # Append non-empty lines to the current value list
            current_layer[value_type].append(stripped_line)
        else:
            # Stop capturing when an empty line is encountered
            capture_value = False

# Append the last layer if it exists
if current_layer:
    parsed_layers_data.append(current_layer)

# Create a DataFrame from the parsed data
df_parsed_layers = pd.DataFrame(parsed_layers_data)

# Convert lists of strings into single strings, separated by commas, for readability
df_parsed_layers["Output"] = df_parsed_layers["Output"].apply(lambda x: "\n".join(x) if x else None)
df_parsed_layers["Weight"] = df_parsed_layers["Weight"].apply(lambda x: "\n".join(x) if x else None)
df_parsed_layers["Bias"] = df_parsed_layers["Bias"].apply(lambda x: "\n".join(x) if x else None)
df_parsed_layers["Weight Gradient"] = df_parsed_layers["Weight Gradient"].apply(lambda x: "\n".join(x) if x else None)
df_parsed_layers["Bias Gradient"] = df_parsed_layers["Bias Gradient"].apply(lambda x: "\n".join(x) if x else None)

df_parsed_layers.drop_duplicates(inplace=True)
df_parsed_layers.to_csv('parsed_data.csv')

