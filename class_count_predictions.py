#This function counts the number of predictions for each class in a batch prediction and logs the results.
def main():
    # store all the content
    sumword = ''
    # count the total number of images
    sumImage = 0
    # List of predicted class names (modify)
    classesList = ['calling','clapping','cycling','dancing','drinking']

    # Open the file containing prediction results for reading
    file = open("data_set/kaggle-human-action-recognition/predict_records/predict_results.txt", "r+")

    for line in file:
        data = line.strip('\n') # Remove newline characters
        sumword += data # Append the line content to sumword
        sumImage = sumImage + 1 # Increment the image count

    file.write("Total number of images:" + str(sumImage))

    for i in range(len(classesList)):
        name = classesList[i]
        count = sumword.count(str(name)) # Count the occurrences of the current class
        file.write(str(name) + ":"+ str(count) + "\n")

    file.close()


if __name__ == '__main__':
    main()
