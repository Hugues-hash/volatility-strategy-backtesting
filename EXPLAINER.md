# A plain English walkthrough

This is the long version of the README, written for someone who has not spent years
around markets. If a term shows up that you have not met, it gets explained here. By
the end you should be able to say what the project does, why each piece is there, and
what the results actually mean. There is a glossary at the bottom for quick lookups.

---

## 1. The one paragraph version

Stock prices jump around. Some days barely move, some days lurch. How much a price
tends to move is called its volatility, and volatility is useful to predict: it tells
you how big a position to take and how much you could lose. This project builds six
different volatility forecasts, then asks three separate questions about them. Which
one predicts best? Which one, used to size a trade, makes the best risk adjusted
return? And which one, used as a risk gauge, most honestly measures the danger? The
punchline is that the answers are three different models. A good forecast, a good
trade, and a good risk number are not the same thing.

---

## 2. What volatility is, and why forecast it

Volatility is the size of the wiggles, not the direction. A calm market and a crashing
market can both be volatile; volatility does not care whether prices go up or down,
only how far they travel.

Two facts make it worth forecasting. First, it clusters. Quiet days follow quiet days,
and once things get wild they stay wild for a while. That means yesterday tells you
something real about tomorrow. Second, it drives the two decisions every investor
makes. If you know tomorrow will be turbulent, you hold less, because the same
position now carries more risk. And if a regulator asks how much you could lose
tomorrow, the honest answer depends almost entirely on volatility.

So the whole project hangs off one number that you cannot even see directly, which
brings us to the first real problem.

---

## 3. Measuring volatility when you cannot see it

You never observe tomorrow's volatility. You only observe the price. So before
forecasting anything, you need a way to measure how volatile a day *was*, after the
fact, to have something to aim at and to score against.

The naive measure is just the size of the day's return. If SPY moved 2 percent, call
that the day's volatility. The problem is that a single number is extremely noisy. A
day can end flat after swinging wildly all afternoon, and the close to close return
would call it calm.

The better measure used here is the **Garman-Klass estimator**. Instead of only the
close, it uses the whole daily bar: the open, the high, the low and the close. A day
that ranged widely gets marked as volatile even if it closed where it started. Using
the range this way is about five times more accurate than the naive measure, which is
why it is standard in the research.

One wrinkle. The daily range runs from the open to the close, so it never captures the
gap between last night's close and this morning's open. Real overnight news lives in
that gap. That makes the Garman-Klass number sit a little below the true close to
close volatility that a strategy actually trades on. The fix in this project is a
single scaling constant, worked out on the first two years of data only, that lifts
the estimate onto the traded scale. It is fitted before the test period starts, so it
is not cheating by peeking at the future.

---

## 4. The six forecasts

All six try to predict tomorrow's volatility from what is known today. They differ in
how much machinery they bring. Here they are, simplest first.

**Rolling.** Take the standard deviation of the last 21 days of returns and call that
your forecast. A one line moving average. It is the honest baseline: if a fancy model
cannot beat this, the machinery is not earning its keep.

**EWMA** (exponentially weighted moving average). Like the rolling window, but instead
of treating the last 21 days equally and everything older as zero, it fades old data
out smoothly, giving more weight to recent days. This is the RiskMetrics method that
banks used for years. The one setting, lambda, controls how fast the memory fades.

**GARCH.** The workhorse of volatility modelling. Its full name is a mouthful, but the
idea is friendly: tomorrow's variance is a mix of a long run average, yesterday's
variance, and yesterday's surprise. When a big move lands, GARCH raises its forecast
and then lets it decay back down over the following days. That decaying response is
exactly the clustering we saw in the data.

**GJR-GARCH** and **EGARCH.** Two variations on GARCH that add one important twist. In
stock markets, a fall tends to raise volatility more than a rise of the same size does.
Fear moves faster than greed. Plain GARCH treats up and down moves the same; GJR and
EGARCH let a drop count for more. This is called the leverage effect, and it turns out
to matter.

**Random Forest.** The machine learning entry. A Random Forest is a crowd of simple
decision trees whose votes are averaged. Here it is fed three summaries of recent
volatility, the **HAR features**: volatility over the last day, the last week, and the
last month. The HAR idea is that different investors care about different horizons, so
a good forecast blends all three. The forest learns how to combine them.

A quick note on what is *not* here. There is no giant neural network and no attempt to
squeeze out the last decimal of accuracy. Small, controlled, well understood models
make a portfolio project you can actually explain, and explaining it is half the point.

---

## 5. Scoring a forecast fairly

Two ideas keep the scoring honest.

**No peeking (walk forward testing).** It is easy to look good if you are allowed to
see the answers. So every model is only ever trained on the past and tested on days it
has never seen. The window of training data rolls forward through time, always ending
before the day being predicted. This mimics real life, where you forecast tomorrow
knowing only up to today.

**The loss functions.** To turn "was the forecast close" into a number, we use three:

- **MSE**, mean squared error, the average of the squared misses. Simple, but it lets
  a few huge misses dominate.
- **MAE**, mean absolute error, the average size of the misses. Steadier.
- **QLIKE**, the one we lead with. It is designed for exactly this problem, where the
  thing you are scoring against (realised volatility) is itself a noisy estimate. It
  also punishes *under* prediction harder than over prediction. For a risk model that
  is the right bias: being caught with too little cushion is worse than carrying a bit
  too much.

Lower is better for all three. And because a smaller number is not always a real
difference, there is one more test.

**Diebold-Mariano.** This asks whether the gap in accuracy between two forecasts is big
enough to be real, or whether it could just be the luck of these particular years. It
looks at the day by day difference in their losses and returns a p value. A small p
value (below 0.05, say) means the difference is very unlikely to be chance. In this
project it confirms that GJR-GARCH's win in accuracy is real, not noise.

---

## 6. Turning a forecast into a strategy

A forecast that changes nothing is just trivia. So each forecast is put to work sizing
a position, using a rule called **volatility targeting**.

Pick a level of risk you are comfortable running, say 10 percent volatility a year.
Each day, look at tomorrow's forecast. If it is calm, hold more of the asset. If it is
stormy, hold less. Size the position so its forecast volatility equals your target:

    weight = target volatility / forecast volatility

When markets are quiet the weight rises (up to a cap so it never takes on wild
leverage), and when they get scary the weight falls, money moving into cash. The
effect is a smoother ride: you automatically pull back before storms and lean in
during calm. This is a real technique that real funds use, not a toy.

The benchmark it is measured against is **buy and hold**: put everything in the asset
on day one and never touch it. If volatility targeting cannot beat just sitting still,
it is not worth the effort or the trading costs.

Speaking of costs, the backtest charges 5 basis points (0.05 percent) every time the
position changes size. This matters more than it sounds, and it is where one of the
findings comes from.

---

## 7. How to read a strategy's scorecard

Once a strategy has run, these are the numbers that describe it. None of them is hard
once you have the intuition.

**Annual return.** The compound growth rate. Not the average of the daily returns, but
the rate that actually turns your starting pound into your ending pound over a year.

**Annual volatility.** How much the strategy itself bounces around, scaled to a yearly
figure. Lower means a calmer ride.

**Sharpe ratio.** The headline number, and the one to really understand. It is return
divided by volatility: how much reward you got for each unit of risk you took. A Sharpe
of 0.6 is ordinary for a stock index. A Sharpe of 0.8 is good. Doubling your return by
doubling your risk does not improve your Sharpe, which is the whole point: it rewards
smart risk, not just more of it.

**Sortino ratio.** Sharpe's cousin. Some volatility is upside, and nobody complains
about big up days. Sortino only counts downside moves as risk, so it rewards a strategy
that is choppy on the way up but gentle on the way down.

**Maximum drawdown.** The worst peak to trough fall the strategy ever suffered. If it
climbed to a high, then dropped 20 percent before recovering, the max drawdown is -20
percent. This is the number that tests your nerve. Buy and hold SPY lost more than half
its value in 2008, a -51 percent drawdown, and most people who felt that in real time
sold at the bottom.

**Calmar ratio.** Annual return divided by that worst drawdown. Return earned per unit
of worst case pain. It rewards strategies that grow without ever scaring you badly.

**Turnover.** How much trading the strategy does. High turnover means high costs, which
is exactly why the most accurate forecast does not win the trading contest here: it
reacts to every twitch, trades constantly, and pays for it.

---

## 8. Value at Risk, and how to check it

The third use of a volatility forecast is measuring danger, and the standard tool is
**Value at Risk**, or VaR.

A 99 percent one day VaR is a loss line. It is the loss you should breach only about
one day in a hundred. If your 99 percent VaR is 3 percent, you are saying "on a normal
day I will not lose more than 3 percent, and I only expect to be wrong about 1 percent
of the time." Building it from a volatility forecast is direct: a bigger forecast
means a wider loss line. Multiply the forecast by a number that depends on how confident
you want to be, and that is your VaR.

The subtle part is the shape of the tail. The simplest assumption is that returns
follow a normal bell curve. Markets do not. They have **fat tails**: extreme days
happen far more often than a bell curve predicts. So a VaR built on the normal
assumption is breached too often, because it did not budget for enough disasters. The
fix is the **Student-t** distribution, which is bell shaped in the middle but with
heavier tails that expect more extreme days. Swapping the normal for a Student-t widens
the loss line and fixes most of the problem.

Now, how do you check whether a VaR model is any good? You backtest it, and there are
two things to test.

**Kupiec test (the count).** Over the whole history, count the days the loss was worse
than the VaR line. If your 99 percent VaR is right, that should happen about 1 percent
of the time. Too many breaches and the model is too optimistic; far too few and it is
wastefully cautious. Kupiec turns "is the breach count about right" into a p value.

**Christoffersen test (the clustering).** Getting the count right is not enough. You
also want the breaches spread out. A model that is calm for years and then breaches
five days in a row during a crisis is dangerous, even if the total count looks fine,
because it fails exactly when you need it. The Christoffersen test checks whether a
breach today makes a breach tomorrow more likely. If breaches cluster, it fails.

A model that passes both, the right number of breaches and no clustering, is one a risk
team could actually sign off. In this project, GJR-GARCH with a Student-t tail is the
only one that gets there for the 99 percent VaR.

---

## 9. The three findings, and why they disagree

Everything above sets up one result. The same six forecasts were put through three
different contests, and they produced three different winners.

**Contest 1, accuracy.** GJR-GARCH forecasts volatility most accurately, and
Diebold-Mariano confirms the win is real. The models that respect the leverage effect
do best, and the plain rolling window is last. Fancy machinery earns its keep here.

**Contest 2, the trade.** Now the simple models win. Volatility targeting works for all
of them, halving the drawdown and lifting the Sharpe from 0.64 to as high as 0.77. But
the best forecaster, GJR-GARCH, reacts to every wobble, trades twice as much, and gives
its edge back in costs. The steadier rolling and EWMA forecasts end up with the best
risk adjusted return. Being right more often did not translate into making more money.

**Contest 3, the risk model.** GJR-GARCH comes back to win. As a 99 percent VaR it is
the only model that passes both backtests: the right number of breaches, and no
clustering. The simple models breach in clumps during crises, which is the worst
possible time for a risk gauge to fail.

So which model is best? The question is not answerable on its own. It depends entirely
on the job. If you want the sharpest forecast, or the most honest risk number, use
GJR-GARCH. If you want to actually trade with low costs, keep it simple. This is the
real lesson, and it is a genuinely useful one: you validate a model *for the use it is
put to*, not in the abstract. A number that looks best on a leaderboard can be the
wrong choice in practice.

---

## 10. Does it hold up? The FTSE check

One index could be a fluke, so the whole pipeline was rerun on the UK's FTSE 100.

Most of the story survived. The GARCH family still forecasts best and rolling still
comes last. GJR-GARCH with a Student-t tail is still a valid 99 percent VaR, and the
simple models still clump their breaches. Volatility targeting still cuts the drawdown.

One thing changed, and it is honest to report it. On the FTSE, volatility targeting did
*not* improve the Sharpe ratio. The reason is simple once you see it: the FTSE barely
rose over these years, around 3.6 percent a year, so pulling money out during calm
spells cost more in missed return than it saved in risk. The risk control still worked,
the drawdowns were still smaller, but the trade off no longer paid on that index.

That is worth more than a result that always works. It shows the strategy's benefit
depends on the asset, which is the kind of caveat that separates a careful piece of
work from a sales pitch.

---

## 11. What this project does not claim

A few honest limits, so nobody reads too much into it.

It uses daily data, not the minute by minute data big desks use, so the volatility
estimates are as good as daily prices allow and no better. It trades a single index
against cash, not a whole portfolio. Cash is assumed to earn nothing, which is slightly
harsh. And it does not include slippage or the messier costs of trading at size. None
of this changes the three findings, which are about the *relative* behaviour of the
models, but the absolute return numbers should be read as illustrative, not as a track
record.

---

## 12. Glossary

- **Volatility** — how much a price moves, regardless of direction.
- **Realised volatility** — a measure of how volatile a day actually was, worked out
  after the fact. Here, the Garman-Klass estimator.
- **Garman-Klass estimator** — a volatility measure that uses the day's open, high,
  low and close, steadier than using the close alone.
- **HAR features** — recent volatility summarised over a day, a week and a month.
- **GARCH / GJR-GARCH / EGARCH** — a family of models where today's volatility depends
  on recent volatility and recent surprises. GJR and EGARCH add the leverage effect.
- **Leverage effect** — the tendency of a fall to raise volatility more than a rise.
- **EWMA** — a volatility estimate that fades old data out smoothly.
- **Walk forward testing** — training only on the past and testing on unseen days.
- **QLIKE / MSE / MAE** — ways to score how close a forecast was, lower is better.
- **Diebold-Mariano test** — checks whether a difference in accuracy is real or luck.
- **Volatility targeting** — sizing a position so its forecast volatility hits a chosen
  level; hold less when it is stormy, more when it is calm.
- **Buy and hold** — the do nothing benchmark: fully invested, never traded.
- **Sharpe ratio** — return per unit of risk. Higher is better.
- **Sortino ratio** — like Sharpe, but only downside counts as risk.
- **Maximum drawdown** — the worst peak to trough fall.
- **Calmar ratio** — return per unit of worst drawdown.
- **Turnover** — how much trading a strategy does; more trading means more cost.
- **Value at Risk (VaR)** — a loss line you should only breach with a set small chance.
- **Fat tails** — extreme days happen more often than a normal bell curve predicts.
- **Student-t** — a bell shape with heavier tails, used to model those extreme days.
- **Kupiec test** — checks a VaR has about the right number of breaches.
- **Christoffersen test** — checks the breaches are spread out, not clustered.
