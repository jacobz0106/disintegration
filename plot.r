setwd('/Users/jacobzhu/Repositories/disintegration/Results/BrusselatorSimulation')
library(dplyr)
library(tidyverse)



plot_function = function(method = 'Train_accuracy', type = 'POF'){
  df = read.csv(paste0(method,'_2000_Brusselator2D_interval_11_',type,'.csv'), header = TRUE, comment.char="")
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
  p = ggplot(df, aes(x = as.factor(Train_size), y = accuracy, fill = Sample_method)) +
    geom_boxplot(position = position_dodge(width = 0.8)) +
    ylim(0, 1) +
    labs(
      x = "Training Size",
      y = "Estimation",
      fill = "Sample Method"
    ) +
    theme_bw()
  p
}

setwd('/Users/jacobzhu/Repositories/disintegration/Results/BrusselatorSimulation')
setwd('/Users/jacobzhu/Repositories/disintegration/Results/BrusselatorSimulation/Simulation1/')

plot_box_multi('Estimation_5000_Brusselator2D_interval_20_POF.csv', 'Estimation_5000_Brusselator2D_interval_20_Random.csv',
               'Estimation_Brusselator2D_interval_20_Naive.csv')


plot_box_multi('Estimation_5000_Brusselator2D_interval_20_POF.csv', 'Estimation_5000_Brusselator2D_interval_20_Random.csv',
               'Estimation_Brusselator2D_interval_20_Naive.csv')


plot_box_multi('Train_accuracy_5000_Brusselator2D_interval_20_POF.csv', 'Train_accuracy_5000_Brusselator2D_interval_20_Random.csv',
               'Train_accuracy_5000_Brusselator2D_interval_20_POF.csv')

plot_box('Train_accuracy_5000_Brusselator2D_interval_20_POF.csv')

plot_box('Estimation_Brusselator2D_interval_20_Naive.csv')
plot_box('Estimation_5000_Brusselator2D_interval_20_Random.csv')
plot_box('Estimation_5000_Brusselator2D_interval_20_POF.csv')



# large number simulation
# increase # of intervals to 20,30 how many training points need to be accurate 


# check Q in the event/ peak beta dist. 


# points in the event for POF/Random?


# how many points needed to classify n labels
plot_box_train = function(file){
  df = read.csv(file, header = TRUE, comment.char="")
  print(df)
  names(df) <- gsub("X\\.\\.|X", "", names(df))
  df = pivot_longer(df, cols = everything(),names_to = 'Train_size', values_to = 'accuracy')
  df$Train_size = as.integer(df$Train_size)
  df2 = df %>%
    group_by(Train_size) %>%
    summarise(avg = mean(accuracy), max = max(accuracy), min = min(accuracy))
  print(df2)
  p = ggplot(df, aes(x = Train_size, y = accuracy)) + 
    geom_boxplot(aes(group = Train_size)) + 
    geom_line(data = df2, aes(x = Train_size, y = avg), color = "blue") +
    scale_x_continuous(breaks = unique(df$Train_size)) + 
    ylim(0, 1) +
    labs(
      x = "Training Size",
      y = "Estimation"
    )
  p
}
par(mfrow = c(1,2))
plot_box_train("Train_accuracy_5000_Brusselator2D_interval_20_POF.csv")
plot_box_train("Train_accuracy_5000_Brusselator2D_interval_20_Random.csv")






setwd('/Users/jacobzhu/Repositories/classificationGE/Results/lotkaVolterra/')
plot_box_volta = function(file){
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
    geom_line(data = df2, aes(x = Train_size, y = avg, group = NA))+ ylim(0, 1)+
    labs(
      y = "Estimation"
    ) 
  p
}
setwd('/Users/jacobzhu/Repositories/classificationGE/Results/lotkaVolterra/simulation1/')
plot_box_volta('Estimation_Naive.csv')
plot_box_volta('Predict_accuracy_1000_1000_Random_NN.csv')
plot_box_volta("Predict_accuracy_1000_Random_PPSVMG.csv")
plot_box_volta('Predict_accuracy_1000_1000_POF_NN.csv')
plot_box_volta("Predict_accuracy_1000_1000_POF_PPSVMG.csv")
plot_box_volta('Estimation_1000_1000_Random_PPSVMG.csv')
plot_box_volta('Estimation_1000_1000_Random_NN.csv')
plot_box_volta('Estimation_1000_1000_POF_NN.csv')
plot_box_volta('Estimation_1000_1000_POF_PPSVMG.csv')
setwd('/Users/jacobzhu/Repositories/classificationGE/Results/lotkaVolterra/')
plot_box_volta('Estimation_Naive.csv')
plot_box_volta('Predict_accuracy_1000_1000_Random_NN.csv')
plot_box_volta("Predict_accuracy_1000_1000_Random_PPSVMG.csv")
plot_box_volta('Predict_accuracy_1000_1000_POF_NN.csv')
plot_box_volta("Predict_accuracy_1000_1000_POF_PPSVMG.csv")
plot_box_volta('Estimation_1000_1000_Random_PPSVMG.csv')
plot_box_volta('Estimation_1000_1000_Random_NN.csv')
plot_box_volta('Estimation_1000_1000_POF_NN.csv')
plot_box_volta('Estimation_1000_1000_POF_PPSVMG.csv')
plot_box_volta('Calibrated_Estimation_1000_1000_POF_PPSVMG.csv')



plot_box_multi('simulation2/Calibrated_Estimation_1000_1000_POF_PPSVMG.csv', 'simulation2/Calibrated_Estimation_1000_1000_Random_PPSVMG.csv', 'Estimation_Naive.csv')

plot_box_multi('simulation2/Estimation_1000_1000_POF_PPSVMG.csv', 'simulation2/Estimation_1000_1000_Random_PPSVMG.csv', 'Estimation_Naive.csv')


plot_box_multi('simulation2/Estimation_1000_1000_POF_NN.csv', 'simulation2/Estimation_1000_1000_Random_NN.csv', 'Estimation_Naive.csv')

plot_box_multi('simulation1/Predict_accuracy_1000_Random_PPSVMG.csv', 'simulation1/Predict_accuracy_1000_POF_NN.csv','simulation1/Predict_accuracy_1000_POF_PPSVMG.csv')


plot_box_multi('Estimation_1000_1000_POF_PPSVMG.csv', 'Calibrated2_Estimation_1000_1000_Random_PPSVMG.csv', 'Estimation_Naive.csv')


# refrence on post hoc calibration/ training efficiency


plot_box_multi = function(filePOF,fileRandom,fileEnu){
  df_POF = read_df(filePOF)
  df_Random = read_df(fileRandom)
  df_Enu = read_df(fileEnu)
  df_POF$Sample_method = 'POF-PPSVMG'
  df_Random$Sample_method = 'Random_PPSVMG'
  df_Enu$Sample_method = 'Naice Monte-Carlo'
  df = bind_rows(df_POF,df_Random,df_Enu)
  df = df %>%
    group_by(Sample_method, Train_size) %>%
    mutate(
      mean_acc = mean(accuracy, na.rm = TRUE),
      sd_acc = sd(accuracy, na.rm = TRUE),
      z_score = abs((accuracy - mean_acc) / sd_acc)
    ) %>%
    arrange(desc(z_score), .by_group = TRUE) %>%
    slice(-(1:2))
  
  df2 = df %>%
    group_by(Sample_method, Train_size) %>%
    summarise(
      avg = mean(accuracy, na.rm = TRUE),
      max = max(accuracy, na.rm = TRUE),
      min = min(accuracy, na.rm = TRUE),
      .groups = 'drop'
    )
  p = ggplot(df, aes(x = as.factor(Train_size), y = accuracy, fill = Sample_method)) +
    geom_boxplot(position = position_dodge(width = 0.8)) +
    ylim(0, 0.5) +
    labs(
      x = "Training Size",
      y = "Estimation",
      fill = "Sample Method"
    ) +
    theme_bw()
  p
}


plot_box_multi_NN = function(filePOF,fileRandom,fileEnu){
  df_POF = read_df(filePOF)
  df_Random = read_df(fileRandom)
  df_Enu = read_df(fileEnu)
  df_POF$Sample_method = 'POF-NN'
  df_Random$Sample_method = 'Random_NN'
  df_Enu$Sample_method = 'Naice Monte-Carlo'
  df = bind_rows(df_POF,df_Random,df_Enu)
  # Find and remove the two most extreme points by standard deviation
  df = df %>%
    group_by(Sample_method, Train_size) %>%
    mutate(
      mean_acc = mean(accuracy, na.rm = TRUE),
      sd_acc = sd(accuracy, na.rm = TRUE),
      z_score = abs((accuracy - mean_acc) / sd_acc)
    ) %>%
    arrange(desc(z_score), .by_group = TRUE) %>%
    slice(-(1:2))
    
  df2 = df %>%
    group_by(Sample_method, Train_size) %>%
    summarise(
      avg = mean(accuracy, na.rm = TRUE),
      max = max(accuracy, na.rm = TRUE),
      min = min(accuracy, na.rm = TRUE),
      .groups = 'drop'
    )
  p = ggplot(df, aes(x = as.factor(Train_size), y = accuracy, fill = Sample_method)) +
    geom_boxplot(position = position_dodge(width = 0.8)) +
    ylim(0, 1) +
    labs(
      x = "Training Size",
      y = "Estimation",
      fill = "Sample Method"
    ) +
    theme_bw()
  p
}

plot_box_multi('Calibrated2_Estimation_1000_1000_POF_PPSVMG.csv', 'Calibrated2_Estimation_1000_1000_Random_PPSVMG.csv', 'Estimation_Naive.csv')
plot_box_multi('Estimation_1000_1000_POF_PPSVMG.csv', 'Estimation_1000_1000_Random_PPSVMG.csv', 'Estimation_Naive.csv')
plot_box_multi_NN('Estimation_1000_1000_POF_NN.csv', 'Estimation_1000_1000_Random_NN.csv', 'Estimation_Naive.csv')


# Two samples
x = c(2.4, 2.6, 2.8, 2.9, 3.1)
y = c(3.5, 3.7, 3.8, 4.0, 4.2)

# Welch's t-test
result = t.test(x, y, var.equal = FALSE)

print(result$p.value)



plot_mean_test = function(filePOF,fileRandom,fileEnu){
  df_POF = read_df(filePOF)
  df_Random = read_df(fileRandom)
  df_Enu = read_df(fileEnu)
  df_POF$Sample_method = 'POF-PPSVMG'
  df_Random$Sample_method = 'Random_PPSVMG'
  df_Enu$Sample_method = 'Naive Monte-Carlo'
  df = bind_rows(df_POF,df_Random,df_Enu)
  df = df %>%
    group_by(Sample_method, Train_size) %>%
    mutate(
      mean_acc = mean(accuracy, na.rm = TRUE),
      sd_acc = sd(accuracy, na.rm = TRUE),
      z_score = abs((accuracy - mean_acc) / sd_acc)
    ) %>%
    arrange(desc(z_score), .by_group = TRUE) %>%
    slice(-(1:2))
  
  # Perform Welch's t-test comparisons
  pvals = df %>%
    filter(Sample_method %in% c("POF-PPSVMG", "Random_PPSVMG", "Naive Monte-Carlo")) %>%
    group_by(Train_size) %>%
    summarise(
      pval_2_vs_1 = t.test(accuracy[Sample_method == "Naive Monte-Carlo"],
                           accuracy[Sample_method == "Random_PPSVMG"],
                           var.equal = FALSE)$p.value,
      
      pval_3_vs_1 = t.test(accuracy[Sample_method == "Naive Monte-Carlo"],
                           accuracy[Sample_method == "POF-PPSVMG"],
                           var.equal = FALSE)$p.value,
      .groups = 'drop'
    )
  
  print(pvals)
  
  # Plot the p-values
  p = ggplot(pvals, aes(x = Train_size)) +
    geom_line(aes(y = pval_2_vs_1, color = "Naive Monte-Carlo vs Random_PPSVMG")) +
    geom_point(aes(y = pval_2_vs_1, color = "Naive Monte-Carlo vs Random_PPSVMG")) +  # Add points for 2 vs 1
    geom_line(aes(y = pval_3_vs_1, color = "Naive Monte-Carlo vs POF-PPSVMG")) +
    geom_point(aes(y = pval_3_vs_1, color = "Naive Monte-Carlo vs POF-PPSVMG")) +  # Add points for 3 vs 1
    geom_hline(yintercept = 0.05, linetype = "dashed", color = "black") + # significance level line
    scale_y_continuous(trans = "log10") + # Optional: log scale for p-values
    labs(
      x = "Training Size",
      y = "p-value (log scale)",
      color = "Comparison",
      title = "Welch's t-test p-values across Training Sizes"
    )  + 
    theme_bw()
  p
  }
  


plot_var_test = function(filePOF,fileRandom,fileEnu){
  df_POF = read_df(filePOF)
  df_Random = read_df(fileRandom)
  df_Enu = read_df(fileEnu)
  df_POF$Sample_method = 'POF-PPSVMG'
  df_Random$Sample_method = 'Random_PPSVMG'
  df_Enu$Sample_method = 'Naive Monte-Carlo'
  df = bind_rows(df_POF,df_Random,df_Enu)
  df = df %>%
    group_by(Sample_method, Train_size) %>%
    mutate(
      mean_acc = mean(accuracy, na.rm = TRUE),
      sd_acc = sd(accuracy, na.rm = TRUE),
      z_score = abs((accuracy - mean_acc) / sd_acc)
    ) %>%
    arrange(desc(z_score), .by_group = TRUE) %>%
    slice(-(1:2))
  
  # Perform Welch's t-test comparisons
  pvals = df %>%
    filter(Sample_method %in% c("POF-PPSVMG", "Random_PPSVMG", "Naive Monte-Carlo")) %>%
    group_by(Train_size) %>%
    summarise(
      pval_2_vs_1 = var.test(accuracy[Sample_method == "Naive Monte-Carlo"],
                           accuracy[Sample_method == "Random_PPSVMG"],
                           alternative = "greater")$p.value,
      
      pval_3_vs_1 = var.test(accuracy[Sample_method == "Naive Monte-Carlo"],
                           accuracy[Sample_method == "POF-PPSVMG"],
                           alternative = "greater")$p.value,
      .groups = 'drop'
    )
  
  print(pvals)
  
  # Plot the p-values
  p = ggplot(pvals, aes(x = Train_size)) +
    geom_line(aes(y = pval_2_vs_1, color = "Naive Monte-Carlo vs Random_PPSVMG")) +
    geom_point(aes(y = pval_2_vs_1, color = "Naive Monte-Carlo vs Random_PPSVMG")) +  # Add points for 2 vs 1
    geom_line(aes(y = pval_3_vs_1, color = "Naive Monte-Carlo vs POF-PPSVMG")) +
    geom_point(aes(y = pval_3_vs_1, color = "Naive Monte-Carlo vs POF-PPSVMG")) +  # Add points for 3 vs 1
    geom_hline(yintercept = 0.05, linetype = "dashed", color = "black") + # significance level line
    scale_y_continuous(trans = "log10") + # Optional: log scale for p-values
    labs(
      x = "Training Size",
      y = "p-value (log scale)",
      color = "Comparison",
      title = "Welch's t-test p-values across Training Sizes"
    )  + 
    theme_bw()
  p
}
plot_mean_test('Estimation_1000_1000_POF_PPSVMG.csv', 'Estimation_1000_1000_Random_PPSVMG.csv', 'Estimation_Naive.csv')
plot_var_test('Estimation_1000_1000_POF_PPSVMG.csv', 'Estimation_1000_1000_Random_PPSVMG.csv', 'Estimation_Naive.csv')


for(k in 0:365){
  print(1 - sum(dbinom(0:k,365,prob = 0.5)))
}

print(1 - sum(dbinom(0:200,365,prob = 0.5)))
  

# --------------------------- t - test plot


library(ggplot2)
library(dplyr)
library(patchwork)
guides_build_mod <- function (guides, theme){
  legend.spacing.y <- calc_element("legend.spacing.y", theme)  # modified by me
  legend.spacing.x <- calc_element("legend.spacing.x", theme)  # modified by me
  legend.box.margin <- calc_element("legend.box.margin", theme) %||% 
    margin()
  widths <- exec(unit.c, !!!lapply(guides, gtable_width))
  heights <- exec(unit.c, !!!lapply(guides, gtable_height))
  just <- valid.just(calc_element("legend.box.just", theme))
  xjust <- just[1]
  yjust <- just[2]
  vert <- identical(calc_element("legend.box", theme), "horizontal")
  guides <- lapply(guides, function(g) {
    editGrob(g, vp = viewport(x = xjust, y = yjust, just = c(xjust, 
                                                             yjust), height = if (vert) 
                                                               heightDetails(g)
                              else 1, width = if (!vert) 
                                widthDetails(g)
                              else 1))
  })
  guide_ind <- seq(by = 2, length.out = length(guides))
  sep_ind <- seq(2, by = 2, length.out = length(guides) - 1)
  if (vert) {
    heights <- max(heights)
    if (length(widths) != 1) {
      w <- unit(rep_len(0, length(widths) * 2 - 1), "mm")
      w[guide_ind] <- widths
      w[sep_ind] <- legend.spacing.x
      widths <- w
    }
  }
  else {
    widths <- max(widths)
    if (length(heights) != 1) {
      h <- unit(rep_len(0, length(heights) * 2 - 1), "mm")
      h[guide_ind] <- heights
      h[sep_ind] <- legend.spacing.y
      heights <- h
    }
  }
  widths <- unit.c(legend.box.margin[4], widths, legend.box.margin[2])
  heights <- unit.c(legend.box.margin[1], heights, legend.box.margin[3])
  guides <- gtable_add_grob(gtable(widths, heights, name = "guide-box"), 
                            guides, t = 1 + if (!vert) 
                              guide_ind
                            else 1, l = 1 + if (vert) 
                              guide_ind
                            else 1, name = "guides")
  gtable_add_grob(guides, element_render(theme, "legend.box.background"), 
                  t = 1, l = 1, b = -1, r = -1, z = -Inf, clip = "off", 
                  name = "legend.box.background")
}



plot_pval_comparison <- function(filePOF, fileRandom, fileEnu) {
  # Read and prepare data
  df_POF <- read_df(filePOF)
  df_Random <- read_df(fileRandom)
  df_Enu <- read_df(fileEnu)
  
  df_POF$Sample_method <- 'POF-PPSVMG'
  df_Random$Sample_method <- 'Random_PPSVMG'
  df_Enu$Sample_method <- 'Naive Monte-Carlo'
  
  df <- bind_rows(df_POF, df_Random, df_Enu) %>%
    group_by(Sample_method, Train_size) %>%
    mutate(
      mean_acc = mean(accuracy, na.rm = TRUE),
      sd_acc = sd(accuracy, na.rm = TRUE),
      z_score = abs((accuracy - mean_acc) / sd_acc)
    ) %>%
    arrange(desc(z_score), .by_group = TRUE) %>%
    slice(-(1:2))
  
  # Welch t-test p-values
  pvals_ttest <- df %>%
    group_by(Train_size) %>%
    summarise(
      `Naive vs Random` = t.test(accuracy[Sample_method == "Naive Monte-Carlo"],
                                 accuracy[Sample_method == "Random_PPSVMG"],
                                 var.equal = FALSE, alternative = "smaller")$p.value,
      `Naive vs POF` = t.test(accuracy[Sample_method == "Naive Monte-Carlo"],
                              accuracy[Sample_method == "POF-PPSVMG"],
                              var.equal = FALSE, alternative = "smaller")$p.value,
      .groups = 'drop'
    ) %>%
    pivot_longer(cols = starts_with("Naive"), names_to = "Comparison", values_to = "pval")
  
  p1 <- ggplot(pvals_ttest, aes(x = Train_size, y = pval, color = Comparison)) +
    geom_line() +
    geom_point() +
    geom_hline(yintercept = 0.05, linetype = "dashed", color = "black") +
    scale_y_continuous(trans = "log10") +
    labs(
      title = "Welch's t-test p-values",
      x = "Training Size",
      y = "p-value",
      color = "Comparison"
    ) +
    theme_bw()
  
  # Variance test p-values
  pvals_var <- df %>%
    group_by(Train_size) %>%
    summarise(
      `Naive vs Random` = var.test(accuracy[Sample_method == "Naive Monte-Carlo"],
                                   accuracy[Sample_method == "Random_PPSVMG"],
                                   alternative = "greater")$p.value,
      `Naive vs POF` = var.test(accuracy[Sample_method == "Naive Monte-Carlo"],
                                accuracy[Sample_method == "POF-PPSVMG"],
                                alternative = "greater")$p.value,
      .groups = 'drop'
    ) %>%
    pivot_longer(cols = starts_with("Naive"), names_to = "Comparison", values_to = "pval")
  
  p2 <- ggplot(pvals_var, aes(x = Train_size, y = pval, color = Comparison)) +
    geom_line() +
    geom_point() +
    geom_hline(yintercept = 0.05, linetype = "dashed", color = "black") +
    scale_y_continuous(trans = "log10") +
    labs(
      title = "Variance test p-values (alternative:variance of naive > vatiance of surrogate model based estimation)",
      x = "Training Size",
      y = "p-value",
      color = "Comparison"
    ) +
    theme_bw()
  
  # Combine plots with shared legend using patchwork
  print(123)
  # Combine safely with patchwork
  combined_plot <- (p1 / p2) + 
    plot_layout(guides = "collect") & 
    theme(legend.position = "bottom",plot.title = element_text(hjust = 0.5))  # Center the title)
  
  return(combined_plot)
}

environment(guides_build_mod) <- asNamespace('patchwork')
assignInNamespace("guides_build", guides_build_mod, ns = "patchwork")
plot_pval_comparison('Estimation_1000_1000_POF_PPSVMG.csv', 'Estimation_1000_1000_Random_PPSVMG.csv', 'Estimation_Naive.csv')


