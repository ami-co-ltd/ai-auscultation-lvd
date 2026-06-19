library(tidyverse)
library(pROC)
library(ggplot2)
library(tableone)
library(readxl)
library(writexl)
library(boot)
library(dplyr)
set.seed(0)
here::here()

# PQ間隔、QRS幅の中央値を追加 =================================================
internal_data_raw <- read.csv("BNP_imputed_internal_mean.csv")
nk_internal <- read.csv("nk2.internal.csv")

# 有効値のみで中央値を計算（NA と -1 を除外）
feat_summary <- nk_internal %>%
  group_by(ID) %>%
  summarise(
    pq_time = median(pq_time[!is.na(pq_time) & pq_time != -1], na.rm = FALSE),
    qrs_time = median(qrs_time[!is.na(qrs_time) & qrs_time != -1], na.rm = FALSE),
    .groups = "drop"
  )
# 有効値が1件もない場合、median() は numeric(0) を返すので NA になる

# master に結合（feat に存在しない ID は NA になる）
internal_with_feat <- internal_data_raw %>%
  left_join(feat_summary, by = "ID")
# =============================================================================

internal_data <- internal_with_feat %>%
  filter(!is.na(LVEF2_cat) & !is.na(age) & !is.na(sex) & !is.na(BMI) & !is.na(DPC_IHD) & !is.na(DPC_CM) & !is.na(Afib) & !is.na(pq_time) & !is.na(qrs_time))

# 2. 欠測除外後の症例IDを確認
excluded_ids_internal <- setdiff(internal_with_feat$ID, internal_data$ID)

# 3. glm 実行
regression_model <- glm(LVEF2_cat ~ age + sex + BMI + DPC_IHD + DPC_CM + Afib + pq_time + qrs_time + log_BNP_pred, data = internal_with_feat, family = binomial)

# 4. 予測確率
internal_data <- internal_data %>%
  mutate(pred_prob = predict(regression_model, type = "response"))

# 5. ROCとAUC
roc_reg_model <- roc(response = internal_data$LVEF2_cat, predictor = internal_data$pred_prob)
plot(roc_reg_model, print.auc = TRUE)

saveRDS(list(roc = roc_reg_model_external, label = "w/ imputed log BNP, PQ time and QRS time", n = nrow(external_data)), "roc_logBNP_pq_qrs.rds")

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
external_data_raw <- read.csv("BNP_imputed_external_mean.csv")

# PQ間隔、QRS幅の中央値を追加 =================================================
nk_external <- read.csv("nk2.external.csv")

# 有効値のみで中央値を計算（NA と -1 を除外）
feat_summary_external <- nk_external %>%
  group_by(ID) %>%
  summarise(
    pq_time = median(pq_time[!is.na(pq_time) & pq_time != -1], na.rm = FALSE),
    qrs_time = median(qrs_time[!is.na(qrs_time) & qrs_time != -1], na.rm = FALSE),
    .groups = "drop"
  )
# 有効値が1件もない場合、median() は numeric(0) を返すので NA になる

# master に結合（feat に存在しない ID は NA になる）
external_data_raw <- external_data_raw %>%
  left_join(feat_summary_external, by = "ID")
# =============================================================================

external_data <- external_data_raw %>% 
  filter(!is.na(labels) & !is.na(age) & !is.na(sex) & !is.na(BMI) & !is.na(DPC_IHD) & !is.na(DPC_CM) & !is.na(Afib) & !is.na(NT_proBNP))
external_data <- external_data %>%
  mutate(log_NT_proBNP = log(NT_proBNP))

external_data <- external_data %>% 
  mutate(pred_prob = predict(regression_model, newdata = external_data, type = "response"))

# 7. internalのYouden Indexを当てはめる
external_data <- external_data %>% 
  mutate(pred_bin = if_else(pred_prob >= Youden_threshold, 1, 0))

write.csv(external_data, "logistic_output_external_with_imputed_logBNP_PR_QRS.csv", row.names = FALSE)

# 8. 性能評価
roc_reg_model_external <- roc(external_data$LVEF2_cat, external_data$pred_prob)
plot(roc_reg_model_external, print.auc =TRUE)

saveRDS(list(roc = roc_reg_model_external, label = "w/ imputed log BNP, PQ time and QRS time", n = nrow(external_data)), "roc_imputed_logBNP_pq_qrs.rds")

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
feature_cols <- c("log_BNP_pred", "age", "sex", "BMI", "DPC_IHD", "DPC_CM", "Afib", "pq_time", "qrs_time")

# モデル全体の擬似R²
model_full <- glm(LVEF2_cat ~ age + sex + BMI + DPC_IHD + DPC_CM + Afib + pq_time + qrs_time + log_BNP_pred,
                  data = internal_with_feat, family = binomial())
model_null <- glm(LVEF2_cat ~ 1, data = internal_with_feat, family = binomial())

pseudo_r2 <- 1 - logLik(model_full) / logLik(model_null)
cat("McFadden R²:", round(pseudo_r2, 4), "\n")

sapply(feature_cols, function(col) {
  remaining <- setdiff(feature_cols, col)
  formula_reduced <- as.formula(paste("LVEF2_cat ~", paste(remaining, collapse = " + ")))
  model_reduced <- glm(formula_reduced, data = internal_with_feat, family = binomial())
  
  r2_full    <- as.numeric(1 - logLik(model_full) / logLik(model_null))
  r2_reduced <- as.numeric(1 - logLik(model_reduced) / logLik(model_null))
  r2_full - r2_reduced  # その変数を除いたときのR²の低下量
})

summary(model_full)

# 単変量AUROC
sapply(feature_cols, function(col) {
  formula <- as.formula(paste("LVEF2_cat ~", col))
  model <- glm(formula, data = internal_with_feat, family = binomial())
  pred <- predict(model, newdata = external_data, type = "response")
  roc_obj <- pROC::roc(external_data$LVEF2_cat, pred, quiet = TRUE)
  as.numeric(pROC::auc(roc_obj))
  
  ci_result <- ci(
    roc_obj,
    method = "bootstrap",
    boot.n = 2000
  )
})

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

