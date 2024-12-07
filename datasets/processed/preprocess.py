import pandas as pd

# Load the CSV file from the current directory
# file_name = "./Preprocessed_set_with_twoattack_normals.csv"
file_name=  './Filtered_dataset.csv'

try:
    # Read the CSV file
    data = pd.read_csv(file_name)
    print(len(data))

    # Check if the file is loaded properly
    if not data.empty:
        # Extract the last column
        last_column_name = data.columns[-1]
        last_column_values = data[last_column_name]

        # Get unique values from the last column
        unique_values = last_column_values.unique()

        # Display the unique values
        print(f"Unique values in the last column ('{last_column_name}'):")
        print(unique_values)
    else:
        print("The CSV file is empty.")
except FileNotFoundError:
    print(f"Error: The file '{file_name}' does not exist in the current directory.")
except Exception as e:
    print(f"An error occurred: {e}")


# import pandas as pd

# # Load the CSV file from the current directory
# file_name = './Preprocessed_set1_10000.csv'
# output_file_name = './Filtered_dataset.csv'

# try:
#     # Read the CSV file
#     data = pd.read_csv(file_name)

#     # Check if the file is loaded properly
#     if not data.empty:
#         # Extract the last column
#         last_column_name = data.columns[-1]
#         last_column_values = data[last_column_name]

#         # Get unique values from the last column
#         unique_values = last_column_values.unique()
#         print(f"Unique values in the last column ('{last_column_name}'):")
#         print(unique_values)

#         # Filter rows where the last column has values 0, 1, or 2
#         filtered_data = data[data[last_column_name].isin([0, 1, 2])]

#         # Save the filtered data to a new CSV file
#         filtered_data.to_csv(output_file_name, index=False)
#         print(f"Filtered data saved to '{output_file_name}'.")
#     else:
#         print("The CSV file is empty.")
# except FileNotFoundError:
#     print(f"Error: The file '{file_name}' does not exist in the current directory.")
# except Exception as e:
#     print(f"An error occurred: {e}")
