library(dplyr)
library(readxl)
library(boot)
set.seed(0)
here::here()

calc_sens_spec_ci <- function(actual, predicted, label = "") {
  actual    <- as.integer(actual)
  predicted <- as.integer(predicted)

  TP <- sum(actual == 1 & predicted == 1, na.rm = TRUE)
  TN <- sum(actual == 0 & predicted == 0, na.rm = TRUE)
  FP <- sum(actual == 0 & predicted == 1, na.rm = TRUE)
  FN <- sum(actual == 1 & predicted == 0, na.rm = TRUE)

  # 感度・特異度のWilson信頼区間
  sens_test <- prop.test(TP, TP + FN, conf.level = 0.95, correct = FALSE)
  spec_test <- prop.test(TN, TN + FP, conf.level = 0.95, correct = FALSE)

  cat("=====", label, "=====\n")
  cat("TP:", TP, "TN:", TN, "FP:", FP, "FN:", FN, "\n")
  cat(sprintf("感度:   %.4f (95%% CI: %.4f - %.4f)\n",
              sens_test$estimate, sens_test$conf.int[1], sens_test$conf.int[2]))
  cat(sprintf("特異度: %.4f (95%% CI: %.4f - %.4f)\n",
              spec_test$estimate, spec_test$conf.int[1], spec_test$conf.int[2]))
  cat("\n")

  list(
    label = label,
    TP = TP, TN = TN, FP = FP, FN = FN,
    sens      = sens_test$estimate,
    sens_lower = sens_test$conf.int[1],
    sens_upper = sens_test$conf.int[2],
    spec      = spec_test$estimate,
    spec_lower = spec_test$conf.int[1],
    spec_upper = spec_test$conf.int[2]
  )
}

# データ読み込み
master <- read.csv("BNP_imputed_external_mean.csv", stringsAsFactors = FALSE)
reader_result   <- read_excel("reader_result.xlsx")

df <- master %>%
  left_join(reader_result, by = "ID") %>%
  filter(!is.na(ge_2))

cat("=== 2人以上 ===\n")
calc_sens_spec_ci(df$LVEF2_cat, df$ge_2)

cat("=== 3人以上 ===\n")
calc_sens_spec_ci(df$LVEF2_cat, df$ge_3)

cat("=== 4人以上 ===\n")
calc_sens_spec_ci(df$LVEF2_cat, df$ge_4)

proposal_4 <- read.csv("prediction.test.0.add_sss_hr_4.csv")
proposal_2 <- read.csv("prediction.test.0.add_sss_hr_2.csv")

# master に結合（feat に存在しない ID は NA になる）
df <- df %>%
  left_join(
    proposal_4 %>% select(ID, LVEF_le_40_pred_bin_4 = LVEF_le_40_pred_bin, LVEF_le_40_pred_prob_4 = LVEF_le_40_pred_prob),
    by = "ID"
  )
df <- df %>%
  left_join(
    proposal_2 %>% select(ID, LVEF_le_40_pred_bin_2 = LVEF_le_40_pred_bin, LVEF_le_40_pred_prob_2 = LVEF_le_40_pred_prob),
    by = "ID"
  )
calc_sens_spec_ci(df$LVEF2_cat, df$LVEF_le_40_pred_bin_4)

write.csv(df, "100_result.csv", row.names = FALSE)

cat("=== Reader 1 ===\n")
calc_sens_spec_ci(df$LVEF2_cat, df$bin_reader1)
cat("=== Reader 2 ===\n")
calc_sens_spec_ci(df$LVEF2_cat, df$bin_reader2)
cat("=== Reader 3 ===\n")
calc_sens_spec_ci(df$LVEF2_cat, df$bin_reader3)
cat("=== Reader 4 ===\n")
calc_sens_spec_ci(df$LVEF2_cat, df$bin_reader4)
cat("=== Reader 5 ===\n")
calc_sens_spec_ci(df$LVEF2_cat, df$bin_reader5)
cat("=== Reader 6 ===\n")
calc_sens_spec_ci(df$LVEF2_cat, df$bin_reader6)
cat("=== Reader 7 ===\n")
calc_sens_spec_ci(df$LVEF2_cat, df$bin_reader7)

# 正解率に関するMcNemer検定

df_compare <- df %>%
  mutate(
    correct_1 = as.integer(df$LVEF_le_40_pred_bin_4 == df$LVEF2_cat),
    correct_2 = as.integer(df$ge_4 == df$LVEF2_cat)
  )
mcnemar.test(table(df_compare$correct_1, df_compare$correct_2))

df_compare <- df %>%
  mutate(
    correct_1 = as.integer(df$LVEF_le_40_pred_bin_4 == df$LVEF2_cat),
    correct_2 = as.integer(df$ge_3 == df$LVEF2_cat)
  )
mcnemar.test(table(df_compare$correct_1, df_compare$correct_2))

df_compare <- df %>%
  mutate(
    correct_1 = as.integer(df$LVEF_le_40_pred_bin_4 == df$LVEF2_cat),
    correct_2 = as.integer(df$ge_2 == df$LVEF2_cat)
  )
mcnemar.test(table(df_compare$correct_1, df_compare$correct_2))

# 4部位モデルのaccuracy CI
correct_1 <- sum(df$LVEF_le_40_pred_bin_4 == df$LVEF2_cat)
n <- nrow(df)
acc1_test <- prop.test(correct_1, n, conf.level = 0.95, correct = FALSE)
cat(sprintf("4部位モデル accuracy: %.4f (%.4f-%.4f)\n",
            acc1_test$estimate, acc1_test$conf.int[1], acc1_test$conf.int[2]))
# k=4のaccuracy CI
correct_2 <- sum(df$ge_4 == df$LVEF2_cat)
acc4_test <- prop.test(correct_2, n, conf.level = 0.95, correct = FALSE)
cat(sprintf("k=4 accuracy: %.4f (%.4f-%.4f)\n",
            acc4_test$estimate, acc4_test$conf.int[1], acc4_test$conf.int[2]))
# k=3のaccuracy CI
correct_2 <- sum(df$ge_3 == df$LVEF2_cat)
acc3_test <- prop.test(correct_2, n, conf.level = 0.95, correct = FALSE)
cat(sprintf("k=3 accuracy: %.4f (%.4f-%.4f)\n",
            acc3_test$estimate, acc3_test$conf.int[1], acc3_test$conf.int[2]))
# k=2のaccuracy CI
correct_2 <- sum(df$ge_2 == df$LVEF2_cat)
acc2_test <- prop.test(correct_2, n, conf.level = 0.95, correct = FALSE)
cat(sprintf("k=2 accuracy: %.4f (%.4f-%.4f)\n",
            acc2_test$estimate, acc2_test$conf.int[1], acc2_test$conf.int[2]))

# F1スコアの信頼区間

# F1値を計算する関数
calc_f1 <- function(actual, predicted) {
  TP <- sum(actual == 1 & predicted == 1)
  FP <- sum(actual == 0 & predicted == 1)
  FN <- sum(actual == 1 & predicted == 0)
  precision <- TP / (TP + FP)
  recall    <- TP / (TP + FN)
  2 * precision * recall / (precision + recall)
}

# ブートストラップ用の関数
boot_f1_diff <- function(data, indices) {
  d <- data[indices, ]
  f1_1 <- calc_f1(d$LVEF2_cat, d$ge_4)
  f1_2 <- calc_f1(d$LVEF2_cat, d$LVEF_le_40_pred_bin_4)
  f1_2 - f1_1  # モデル2 - モデル1
}

# ブートストラップ実行
boot_result <- boot(df, boot_f1_diff, R = 2000)

# 信頼区間
boot_ci <- boot.ci(boot_result, type = "perc")
print(boot_ci)
print(boot_result$t0)

# F1値を計算する関数
calc_f1 <- function(actual, predicted) {
  TP <- sum(actual == 1 & predicted == 1)
  FP <- sum(actual == 0 & predicted == 1)
  FN <- sum(actual == 1 & predicted == 0)
  precision <- TP / (TP + FP)
  recall    <- TP / (TP + FN)
  2 * precision * recall / (precision + recall)
}

# ブートストラップでF1のCIを出す関数
f1_ci <- function(df, pred_col, label = "") {
  boot_f1 <- function(data, indices) {
    d <- data[indices, ]
    calc_f1(d$LVEF2_cat, d[[pred_col]])
  }

  boot_result <- boot(df, boot_f1, R = 2000)
  ci <- boot.ci(boot_result, type = "perc")
  
  cat(sprintf("%s F1: %.4f (%.4f-%.4f)\n",
              label,
              boot_result$t0,
              ci$percent[4],
              ci$percent[5]))
}

# 各モデルのF1とCI
f1_ci(df, "LVEF_le_40_pred_bin_4", "4部位モデル")
f1_ci(df, "ge_2", "k=2")
f1_ci(df, "ge_3", "k=3")
f1_ci(df, "ge_4", "k=4")
