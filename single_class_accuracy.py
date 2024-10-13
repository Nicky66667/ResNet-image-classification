# This function calculates the number of correct predictions for a specific class in a batch prediction.

def main():
    # Initialize an empty string to store all prediction results
    sumword = ''

    # The specific class to search for in the predictions (Modify as needed)
    word = "dandelion"

    # Initialize a counter for the total number of images
    sumImage = 0

    # The file should have one prediction result per line
    file = open("data_set/predict_results.txt", "r+")

    for line in file:
        data = line.strip('\n') # Remove newline characters
        sumword += data # Concatenate the prediction results into one string
        sumImage = sumImage + 1 # Increment the image counter for each line in the file

    # Check if the specific class appears in the predictions
    if word in sumword:
        # Write the correct class name
        file.write("correct class:" + word + "\n")
        # Calculate the correctness of predictions for the specified class
        correctness = sumword.count(word) / sumImage
        file.write("total images: " + str(sumImage) + "\n")
        file.write("correct images: " + str(sumword.count(word)) + "\n")
        file.write("correctness: " + str(correctness) + "\n")

    file.close()


if __name__ == '__main__':
    main()
