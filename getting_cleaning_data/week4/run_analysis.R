# Read in data
x_test <- read.table('test/X_test.txt')
y_test <- read.table('test/y_test.txt')
x_train <- read.table('train/X_train.txt')
y_train <- read.table('train/y_train.txt')
sub_test <- read.table('test/subject_test.txt')
sub_train <- read.table('train/subject_train.txt')

#Combine data
X <- rbind(x_train, x_test)
y <- rbind(y_train, y_test)
sub <- rbind(sub_train, sub_test)

# 2), 4)
#Filter columns for mean & std for X
feature_names <- read.table('features.txt', sep=' ')
colnames(feature_names) = c('index', 'name')

filtered_features <- feature_names[grep('.mean.|.std.', feature_names$name),]

X <- X[,filtered_features$index]
colnames(X) = filtered_features$name

colnames(y) <- c('label')

colnames(sub) <- c('subjectid')

label_names <- read.table('activity_labels.txt', sep=' ')
colnames(label_names) = c('index', 'name')

# 3) Convert ints to factors
y <- factor(y$label, labels=as.character(label_names$name))
colnames(y) <- c('label')
# 5) Make tidy data set
library(tidyr)
library(dplyr)

combo <-  cbind(X, y, sub)

means <- combo %>%
    group_by(y, subjectid) %>%
    summarise_all(mean)
colnames(means)[1] = 'activity'

write.table(means, 'activities.txt', row.names=FALSE)






















