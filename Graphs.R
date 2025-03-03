# -------------------------------------------------------------------------------------------------------------------- #
#                                             Statistical Arbitrage Crypto                                             #
#                                                Quantitative Research                                                 #
# -------------------------------------------------------------------------------------------------------------------- #

library(tidyverse)

# Loading Data
cumulative_returns = read.csv('Cumulative Returns.csv')
cumulative_returns$Date = as.Date(cumulative_returns$Date, format="%Y-%m-%d")

# Theme
theme_set(
  theme_grey() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 16),
      axis.line = element_line(linewidth = 1, colour = 'grey80'),
      text = element_text(family = 'CMU Serif', size = 12)
    )
)

# Plot the cumulative returns
ggplot(cumulative_returns, aes(x = Date, y = X0)) +
  geom_area(alpha=0.5) +
  geom_line(linewidth=0.25) +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y") +
  scale_y_continuous(breaks = seq(floor(min(cumulative_returns$X0)-1),
                                  ceiling(max(cumulative_returns$X0)), by = 5)) +
  labs(title = "Portfolio Cumulative Returns", x = "Date", y = "Cumulative Return")


ggsave('cumulative_returns.png', dpi=300, width=10, height=5)