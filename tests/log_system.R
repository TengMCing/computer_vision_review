library(cli)
epochs <- 20
batches <- 1875

metrics <- data.frame(loss = rep(NA, epochs),
                      accuracy = rep(NA, epochs))
viewer <- NULL
for (i in 1:epochs) {
  loss <- 0
  accuracy <- 0
  cli_alert_info("{col_yellow('Epoch')} {col_yellow(i)}/{col_yellow(epochs)}")
  cli_progress_bar(total = batches,
                   format = "{pb_spin} {col_red('Training')} {.val {pb_current}}/{.val {pb_total}} {pb_bar} | {.field ETA:} {pb_eta} | {.field loss:} {sprintf(loss, fmt = '%#.4f')} | {.field accuracy:} {sprintf(accuracy, fmt = '%#.4f')}",
                   format_done = "{col_green(symbol$tick)} {col_green('Trained')} {.val {pb_current}}/{.val {pb_total}} | {sprintf(pb_elapsed_raw, fmt = '%#.2f')}s - {sprintf(pb_elapsed_raw/batches, fmt = '%#.4f')}s/step | {.field loss:} {sprintf(loss, fmt = '%#.4f')} | {.field accuracy:} {sprintf(accuracy, fmt = '%#.4f')}",
                   clear = FALSE)
  for (j in 1:batches) {
    Sys.sleep(0.005)
    loss <- abs(runif(1))
    accuracy <- abs(runif(1))
    cli_progress_update()
  }
  metrics$loss[i] <- loss
  metrics$accuracy[i] <- accuracy
  if (is.null(viewer)) {
    viewer <- tfruns::view_run_metrics(metrics)
  } else {
    tfruns::update_run_metrics(viewer, metrics)
  }
}
