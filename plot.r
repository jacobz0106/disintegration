setwd('/Users/jacobzhu/Repositories/disintegration/Results/BrusselatorSimulation')
library(dplyr)
library(tidyverse)



plot_function = function(method = 'Train_accuracy', type = 'POF'){
  df = read.csv(paste0(method,'_2000_Brusselator2D_interval_6_',type,'.csv'), header = TRUE, comment.char="")
  print(df)
  names(df) <- gsub("X\\.\\.|X", "", names(df))
  df = pivot_longer(df, 1:7,names_to = 'Train_size', values_to = 'accuracy')
  df$Train_size = as.integer(df$Train_size)
  df = df %>%
    group_by(Train_size) %>%
    summarise(avg = mean(accuracy), max = max(accuracy), min = min(accuracy))
  
  p = ggplot(df, aes(x = Train_size, y = avg)) +  geom_point() + geom_line() + 
    geom_errorbar(aes(ymin=min, ymax=max),width = 0.2 ) +
    ylim(0, 1)
  p
}

plot_function(method = 'Train_accuracy', type = 'POF')
plot_function(method = 'Train_accuracy', type = 'Random')
plot_function(method = 'Estimation', type = 'POF')
plot_function(method = 'Estimation', type = 'Random')

# Naive method

df = read.csv('Estimation_Brusselator2D_interval_11_Naive.csv', header = TRUE, comment.char="")
names(df) <- gsub("X\\.\\.|X", "", names(df))
df = pivot_longer(df, 1:14,names_to = 'Train_size', values_to = 'accuracy')
df$Train_size = as.integer(df$Train_size)
df = df %>%
  group_by(Train_size) %>%
  summarise(avg = mean(accuracy), max = max(accuracy), min = min(accuracy))

p = ggplot(df, aes(x = Train_size, y = avg)) +  geom_point() + geom_line() + 
  geom_errorbar(aes(ymin=min, ymax=max),width = 0.2 ) +
  ylim(0, 1)
p

# how many points needed to classify n labels
plot_box = function(file){
  df = read.csv(file, header = TRUE, comment.char="")
  print(df)
  names(df) <- gsub("X\\.\\.|X", "", names(df))
  df = pivot_longer(df, cols = everything(),names_to = 'Train_size', values_to = 'accuracy')
  df$Train_size = as.integer(df$Train_size)
  df2 = df %>%
    group_by(Train_size) %>%
    summarise(avg = mean(accuracy), max = max(accuracy), min = min(accuracy))
  print(df2)
  p = ggplot(df, aes(x = Train_size, y = accuracy, group=Train_size)) + geom_boxplot()+ 
    geom_line(data = df2, aes(x = Train_size, y = avg, group = NA))+ ylim(0, 0.2)+
    labs(
      y = "Estimation"
    ) 
  p
}
plot_box('Estimation_Brusselator2D_interval_11_Naive.csv')
plot_box('Estimation_2000_Brusselator2D_interval_11_POF.csv')
plot_box('Estimation_2000_Brusselator2D_interval_11_Random.csv')


#
read_df = function(file){
  df = read.csv(file, header = TRUE, comment.char="")
  names(df) <- gsub("X\\.\\.|X", "", names(df))
  df = pivot_longer(df, cols = everything(),names_to = 'Train_size', values_to = 'accuracy')
  df$Train_size = as.integer(df$Train_size)
  return(df)
}
plot_box_multi = function(filePOF,fileRandom,fileEnu){
  df_POF = read_df(filePOF)
  df_Random = read_df(fileRandom)
  df_Enu = read_df(fileEnu)
  df_POF$Sample_method = 'POF'
  df_Random$Sample_method = 'Random'
  df_Enu$Sample_method = 'Enumerate'
  df = bind_rows(df_POF,df_Random,df_Enu)
  df2 = df %>%
    group_by(Sample_method, Train_size) %>%
    summarise(avg = mean(accuracy), max = max(accuracy), min = min(accuracy))
  p = ggplot(df, aes(x = as.factor(Train_size), y = accuracy, color = Sample_method)) + geom_boxplot()+ 
    geom_line(data = df2, aes(x = factor(Train_size), y = avg, group = Sample_method))+ ylim(0, 0.2)+ 
    geom_hline(yintercept= 0.0417) + 
    labs(
      y = "Estimation"
    ) 
  p
}

plot_box_multi('Estimation_2000_Brusselator2D_interval_11_POF.csv', 'Estimation_2000_Brusselator2D_interval_11_Random.csv',
               'Estimation_Brusselator2D_interval_11_Naive.csv')

plot_box('Estimation_Brusselator2D_interval_21_Naive.csv')
# large number simulation
# increase # of intervals to 20,30 how many training points need to be accurate 

plot_box_multi('Estimation_Brusselator2D_interval_11_Naive.csv', 'Estimation_Brusselator2D_interval_31_Naive.csv', 'Estimation_Brusselator2D_interval_61_Naive.csv')

# check Q in the event/ peak beta dist. 

# normalize the SVM
0.68663585**2 +  0.72700155**2


birthday = rep(0, 6)
sum(birthday == 1)
check_birthday = function(birthday){
  for (i in c(1:12) ){
    if(sum(birthday==letters[i]) ==3){
      return(TRUE)
    }else if{
      return(FALSE)
    }
  }
}
total = 0 
df = permutations(12,6,letters[1:12],repeats=TRUE)
lapply(df, check_birthday)




#  a)
# 
n = 20
p = 0.7
total = 0
for (i in 0:7){
  d = dbinom(i, size = n, prob = p)
  total = total + d
}
total 
pbinom(11, size = n, prob = p)

ppois(9, lambda =n*p, lower.tail = TRUE)

factorial(15)/(factorial(8)*factorial(7)) * 0.7^7 * 0.3^8












