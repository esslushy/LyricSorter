# Lyric Sorter
This project was a guided machine learning project using the Sklearn library. It was done in order to gain a better understanding of the steps taken to gather, format, and vectorize the data so that it is understandable by the computer.

# Process
The data was first extracted from a csv file and put into a pandas database.  I then checked if there were any nulls and made sure that the artists names were spelled correct and that there were only 2 unique artists. Then I removed all none alphabetical, apostrophe, or space characters from the lyrics.  I than looked at the amount of repeats and what lyrics were repeated to see if it would have a significant effect on the data:  I decided that it did not have an impact.  I then reduced words to just their stem and removed all stop words in the lyrics.  I then split the vector into train and testing groups in order to get the data for the first training. The models used in this were from Sklearn library and were the Multinomial Naive Bayes model and a Support Vector Machine model.  I than built a cross validation system using Stratified Shuffle Split from Sklearn as it allowed me to specify the size of the test split.

# Results
I found that the Multinomial Naive Bayes besides being the fastest to run was also the most effective model. It had a slight instability as shown in cross validation, but it is so small that it is negligent to the effectiveness of the model.
Multinomial Naive Bayes: 
- 1st Run: 75.93621647741%
- Cross Validation Runs: [77.240879% 76%105339 75%984537 75.791254% 75.54965% ]
Support Vector Machine:
- 1st Run: 60.08697753080454%
- Cross Validation Runs: [58.758154% 58.758154% 58.758154% 58.758154% 58.758154%]

# Thanks
Thanks to Reed Coke and Alex from [Hello World](https://www.helloworldstudio.org/home) for teaching me and guiding me through this project.
