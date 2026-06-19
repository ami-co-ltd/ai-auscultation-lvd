library(mice)
library(dplyr)
library(readxl)
set.seed(0)
here::here()

df_raw <- read_excel("clinicaldata.xlsx")

df_raw <- df_raw %>%
  filter(!is.na(LVEF2_cat) & !is.na(age) & !is.na(sex) & !is.na(BMI) & !is.na(DPC_IHD) & !is.na(DPC_CM) & !is.na(Afib))
df_raw <- df_raw %>%
  mutate(log_BNP = log(BNP))
df_raw <- df_raw %>%
  mutate(log_NT_proBNP = log(NT_proBNP))

glimpse(df_raw)
cat("\n欠損値の数:\n")
print(colSums(is.na(df_raw)))

feature_cols <- c(  # 補完に使うカラム
  "log_BNP",
  "log_NT_proBNP",
  "age",
  "sex",
  "BMI",
  "DPC_IHD",
  "DPC_CM",
  "Afib"
)

# 最終CSVに含めるカラム
passthrough_cols <- c(
  "ID",
  "BNP",
  "NT_proBNP",
  "LVEF2_cat",
  "age",
  "sex",
  "BMI",
  "DPC_IHD",
  "DPC_CM",
  "Afib"
)

# 不要カラムを除く
df <- df_raw %>% select(all_of(feature_cols))
df_passthrough <- df_raw %>% select(all_of(passthrough_cols))

cat("\n抽出後のデータ形状:", nrow(df), "行 x", ncol(df), "列\n")
cat("使用カラム:", paste(feature_cols, collapse = ", "), "\n")

# 変数を factor 型に変換
#df[[target_col]] <- factor(df[[target_col]], levels = c(0, 1))
df <- df %>%
  mutate(across(c(Afib), factor))

# データの形式に合わせて補完方法を指定
method_custom <- make.method(df)
method_custom["log_BNP"] <- "norm"
method_custom["log_NT_proBNP"] <- "norm"

# 妥当性チェック用 ============================================================

# 欠損のない行を特定
obs_idx <- which(!is.na(df$log_BNP))

# 欠損のない行のlog_BNPをNAで埋めたデータを作成
df_masked <- df
df_masked$log_BNP[obs_idx] <- NA

# external補完 ================================================================


df_external_raw <- read_excel("ListenHeart_external_test_data6.xlsx")

df_external_raw <- df_external_raw %>%
  filter(!is.na(LVEF2_cat) & !is.na(age) & !is.na(sex) & !is.na(BMI) & !is.na(DPC_IHD) & !is.na(DPC_CM) & !is.na(Afib))
df_external_raw <- df_external_raw %>%
  mutate(log_BNP = log(BNP))
df_external_raw <- df_external_raw %>%
  mutate(log_NT_proBNP = log(NT_proBNP))

glimpse(df_external_raw)
cat("\n欠損値の数:\n")
print(colSums(is.na(df_external_raw)))

# 不要カラムを除く
df_external <- df_external_raw %>% select(all_of(feature_cols))
df_external_passthrough <- df_external_raw %>% select(all_of(passthrough_cols))

cat("\n抽出後のデータ形状:", nrow(df_external), "行 x", ncol(df_external), "列\n")
cat("使用カラム:", paste(feature_cols, collapse = ", "), "\n")

# 変数を factor 型に変換
df_external <- df_external %>%
  mutate(across(c(Afib), factor))

# MICE実行 ====================================================================

# 元データ（学習用）+ マスクしたデータ（予測用）を結合
df_combined <- bind_rows(df, df_masked[obs_idx, ], df_external)

# ignore ベクトル: 元データ行はFALSE、マスク行はTRUE
ignore_vec <- c(
  rep(FALSE, nrow(df)),             # internal: 学習に使う
  rep(TRUE,  length(obs_idx)),      # internalマスク: 予測のみ
  rep(TRUE,  nrow(df_external))     # external: 予測のみ
)

# mice実行
mice_result_all  <- mice(
  data = df_combined,
  m = 100,
  maxit = 20,
  method = method_custom,
  ignore = ignore_vec,
  seed = 0
)

# 各部分の行範囲
n_internal      <- nrow(df)
n_masked        <- length(obs_idx)
n_external      <- nrow(df_external)

idx_internal <- 1:n_internal
idx_masked   <- (n_internal + 1):(n_internal + n_masked)
idx_external <- (n_internal + n_masked + 1):(n_internal + n_masked + n_external)

# 取り出し
imputed_long_all <- complete(mice_result_all, action = "long", include = FALSE)

# internal補完済み
imputed_internal <- imputed_long_all %>%
  filter(.id %in% idx_internal)

# 妥当性検証用
imputed_masked <- imputed_long_all %>%
  filter(.id %in% idx_masked)

# external補完済み
imputed_external <- imputed_long_all %>%
  filter(.id %in% idx_external)

# internal保存 ================================================================

imputed_internal_mean <- imputed_internal %>%
  group_by(.id) %>%
  summarise(
    across(c(log_BNP, log_NT_proBNP), mean, .names = "{.col}_pred"),
    .groups = "drop"
  )

imputed_internal_mean_full <- bind_cols(df_passthrough, imputed_internal_mean)

write.csv(imputed_internal_mean_full, "BNP_imputed_internal_mean.csv", row.names = FALSE)

# 妥当性検証 ==================================================================

imputed_masked_mean <- imputed_masked %>%
  group_by(.id) %>%
  summarise(log_BNP_pred = mean(log_BNP), .groups = "drop")

# 真の値と比較
true_values <- df$log_BNP[obs_idx]
pred_values <- imputed_masked_mean$log_BNP_pred

ss_res <- sum((true_values - pred_values)^2)
ss_tot <- sum((true_values - mean(true_values))^2)
r_squared <- 1 - ss_res / ss_tot

cat("R²:", round(r_squared, 4), "\n")

plot(true_values, pred_values,
     xlab = "true log_BNP",
     ylab = "mean of imputed value",
     main = paste0("Validation of imputing (R² = ", round(r_squared, 4), ")"))
abline(0, 1, col = "red")
abline(v = log(100), col = "blue", lty = 2)  # 縦線 x = ln(100)
abline(h = log(100), col = "blue", lty = 2)  # 横線 y = ln(100)

# external保存 ================================================================

imputed_external_mean <- imputed_external %>%
  group_by(.id) %>%
  summarise(
    across(c(log_BNP, log_NT_proBNP), mean, .names = "{.col}_pred"),
    .groups = "drop"
  )

external_non_numeric_cols <- imputed_external %>%
  ungroup() %>%
  filter(.imp == 1) %>%
  arrange(.id) %>%
  select(where(~ !is.numeric(.)))

imputed_external_mean <- bind_cols(df_external_passthrough, imputed_external_mean)

write.csv(imputed_external_mean, "BNP_imputed_external_mean.csv", row.names = FALSE)

