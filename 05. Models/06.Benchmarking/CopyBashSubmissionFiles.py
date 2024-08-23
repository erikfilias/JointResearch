# Python script to create copies of a bash script with modified numbers

# List of numbers to replace 150 with
numbers_to_replace = [110, 120, 130, 140]

# Read the original file content
with open("B.Operation_cost/sub_loop_IEEE118_150.sh", "r") as file:
    content = file.read()

# Loop through each number and create a new file with the modified content
for number in numbers_to_replace:
    # Replace the number 150 with the current number
    new_content = content.replace("150", str(number))

    # Create a new filename based on the current number
    new_filename = f"B.Operation_cost/sub_loop_IEEE118_{number}.sh"

    # Write the modified content to the new file
    with open(new_filename, "w") as new_file:
        new_file.write(new_content)

    print(f"Created {new_filename} with number {number}")
