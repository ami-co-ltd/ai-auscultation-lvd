library(tidyverse)
library(pROC)
library(ggplot2)
library(tableone)
library(readxl)
library(writexl)
library(boot)
set.seed(0)
here::here()

internal_data <- read.csv("BNP_imputed_internal_mean.csv")

# 3. glm 実行
regression_model <- glm(LVEF2_cat ~ log_BNP_pred, data = internal_data, family = binomial)

# 4. 予測確率
internal_data <- internal_data %>%
  mutate(pred_prob = predict(regression_model, type = "response"))

# 5. ROCとAUC
roc_reg_model <- roc(response = internal_data$LVEF2_cat, predictor = internal_data$pred_prob)
plot(roc_reg_model, print.auc = TRUE)
Youden_index_model <- coords(roc_reg_model, "best", best.method = "youden", ret = "threshold")
Youden_threshold <- Youden_index_model$threshold
print(Youden_threshold)

# AUROCのブートストラップによる95%信頼区間を取得
auc_ci_reg_model <- ci(
  roc_reg_model,
  method = "bootstrap",
  boot.n = 2000,
  progress = "none"  # ワーニング防止
)

print(auc_ci_reg_model)

# 6. Listen Externalに当てはめる#########################################################################################################################################################
external_data <- read.csv("BNP_imputed_external_mean.csv")

external_data <- external_data %>% 
  mutate(pred_prob = predict(regression_model, newdata = external_data, type = "response"))

# 7. internalのYouden Indexを当てはめる
external_data <- external_data %>% 
  mutate(pred_bin = if_else(pred_prob >= Youden_threshold, 1, 0))

write.csv(external_data, "logistic_output_external_logBNP_only.csv", row.names = FALSE)

# 8. 性能評価
roc_reg_model_external <- roc(external_data$LVEF2_cat, external_data$pred_prob)
plot(roc_reg_model_external, print.auc =TRUE)

saveRDS(list(roc = roc_reg_model_external, label = "w/ imputed log BNP", n = nrow(external_data)), "roc_imputed_logBNP_only.rds")

# AUROCのブートストラップによる95%信頼区間を取得
auc_ci_reg_model_external <- ci(
  roc_reg_model_external,
  method = "bootstrap",
  boot.n = 2000,
  progress = "none"
)

# 表示
print(auc_ci_reg_model_external)


# McFaddenの疑似決定係数
feature_cols <- c("log_BNP_pred")

# モデル全体の擬似R²
model_full <- glm(LVEF2_cat ~ log_BNP_pred,
                  data = internal_data, family = binomial())
model_null <- glm(LVEF2_cat ~ 1, data = internal_data, family = binomial())

pseudo_r2 <- 1 - logLik(model_full) / logLik(model_null)
cat("McFadden R²:", round(pseudo_r2, 4), "\n")

# 提案モデルとのAUROCの差の検定 ===============================================

proposal_4 <- read.csv("prediction.test.0.add_sss_hr_4.csv")
proposal_2 <- read.csv("prediction.test.0.add_sss_hr_2.csv")

# master に結合（feat に存在しない ID は NA になる）
external_with_proposal <- external_data %>%
  left_join(
    proposal_4 %>% select(ID, LVEF_le_40_pred_prob_4 = LVEF_le_40_pred_prob),
    by = "ID"
  ) %>%
  left_join(
    proposal_2 %>% select(ID, LVEF_le_40_pred_prob_2 = LVEF_le_40_pred_prob),
    by = "ID"
  )

roc_logistic <- roc(external_with_proposal$LVEF2_cat, external_with_proposal$pred_prob)
roc_4 <- roc(external_with_proposal$LVEF2_cat, external_with_proposal$LVEF_le_40_pred_prob_4)
roc_2 <- roc(external_with_proposal$LVEF2_cat, external_with_proposal$LVEF_le_40_pred_prob_2)

cat("logistic AUROC:", as.numeric(auc(roc_logistic)), "\n")
cat("4部位 AUROC:", as.numeric(auc(roc_4)), "\n")
cat("2部位 AUROC:", as.numeric(auc(roc_2)), "\n")

# DeLong検定
roc_test_result <- roc.test(roc_logistic, roc_4, method = "delong")
print(roc_test_result)
roc_test_result <- roc.test(roc_logistic, roc_2, method = "delong")
print(roc_test_result)

