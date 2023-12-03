import matplotlib.pyplot as plt
# higher interface for pyplot (not necessary)
import seaborn as sns

# Data for plotting
num_streams = 8
x = list(range(1, num_streams + 1))
# performance data for N = M = 2000 in Matrix-Vector Multiplication in millisecond(ms)
y = [99, 51, 42, 33, 28, 24, 21, 18]

plt.figure(figsize = (8,4))
sns.lineplot(x = x, y = y, marker = 'o') # Line plot with markers
# Alternatively
#plt.plot(x, y, marker = 'o')

# Annotate the Data
for i, val in enumerate(y):
    plt.annotate(val, (x[i], y[i]))

plt.xlabel('Number of Streams')
plt.ylabel('Performance in Millisecond(ms)')
plt.title('Performance Improvements w.r.t. the Number of Streams')
plt.show()
