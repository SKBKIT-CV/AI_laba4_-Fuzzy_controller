import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

roe = np.arange(0, 61, 1)
bad_roe = fuzz.trapmf(roe, [0, 0, 7, 15])
good_roe = fuzz.trapmf(roe, [10, 15, 30, 45])
risky_roe = fuzz.trapmf(roe, [35, 50, 61, 61])

ros = np.arange(0, 31, 1) 
low_ros = fuzz.trapmf(ros, [0, 0, 5, 10])
aver_ros = fuzz.trapmf(ros, [5, 10, 15, 20])
high_ros = fuzz.trapmf(ros, [15, 25, 31, 31])

prob = np.arange(0, 101, 1) 
low_prob = fuzz.trimf(prob, [0, 25, 50])
aver_prob = fuzz.trimf(prob, [25, 50, 75])
high_prob = fuzz.trimf(prob, [50, 75, 101])

fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(7, 7))

ax0.plot(roe, bad_roe, 'b', linewidth=1.5, label='Bad')
ax0.plot(roe, good_roe, 'r', linewidth=1.5, label='Optimal')
ax0.plot(roe, risky_roe, 'y', linewidth=1.5, label='Precarious')
ax0.set_ylabel('Degree of membership')
ax0.set_xlabel('ROE')
ax0.legend()

ax1.plot(ros, low_ros, 'b', linewidth=1.5, label='Low')
ax1.plot(ros, aver_ros, 'r', linewidth=1.5, label='Average')
ax1.plot(ros, high_ros, 'y', linewidth=1.5, label='High')
ax1.set_ylabel('Degree of membership')
ax1.set_xlabel('ROS')
ax1.legend()

ax2.plot(prob, low_prob, 'b', linewidth=1.5, label='Low')
ax2.plot(prob, aver_prob, 'r', linewidth=1.5, label='Average')
ax2.plot(prob, high_prob, 'y', linewidth=1.5, label='High')
ax2.set_ylabel('Degree of membership')
ax2.set_xlabel('Probability of Investment')
ax2.legend()

for ax in (ax0, ax1, ax2):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()

# Fuzzy rules
your_roe, your_ros = int(input()), int(input())

roe_lvl_bad = fuzz.interp_membership(roe, bad_roe, your_roe)
roe_lvl_good = fuzz.interp_membership(roe, good_roe, your_roe)
roe_lvl_risky = fuzz.interp_membership(roe, risky_roe, your_roe)

ros_lvl_low = fuzz.interp_membership(ros, low_ros, your_ros)
ros_lvl_aver = fuzz.interp_membership(ros, aver_ros, your_ros)
ros_lvl_high = fuzz.interp_membership(ros, high_ros, your_ros)

# Still fuzzy rules
aver1 = np.fmin(roe_lvl_good, ros_lvl_low)
aver2 = np.fmin(roe_lvl_bad, ros_lvl_aver)
aver3 = np.fmin(roe_lvl_risky, ros_lvl_aver)
aver4 = np.fmin(roe_lvl_bad, ros_lvl_high)
aver5 = np.fmin(roe_lvl_risky, ros_lvl_high)

low1 = np.fmin(roe_lvl_bad, ros_lvl_low)
low2 = np.fmin(roe_lvl_risky, ros_lvl_low)

high1 = np.fmin(roe_lvl_good, ros_lvl_aver)
high2 = np.fmin(roe_lvl_good, ros_lvl_high)

aver_rule = max(aver1, aver2, aver3, aver4, aver5)
low_rule = max(low1, low2)
high_rule = max(high1, high2)

rule1 = np.fmin(aver_rule, aver_prob)
rule2 = np.fmin(low_rule, low_prob)
rule3 = np.fmin(high_rule, high_prob)

prob0 = np.zeros_like(prob)

# Visualization
fig2, ax0 = plt.subplots(figsize=(7, 3))

ax0.fill_between(prob, prob0, rule1, facecolor='r', alpha=0.7)
ax0.plot(prob, aver_prob, 'r', linewidth=0.5, linestyle='--', )
ax0.fill_between(prob, prob0, rule2, facecolor='b', alpha=0.7)
ax0.plot(prob, low_prob, 'b', linewidth=0.5, linestyle='--', )
ax0.fill_between(prob, prob0, rule3, facecolor='y', alpha=0.7)
ax0.plot(prob, high_prob, 'y', linewidth=0.5, linestyle='--', )
ax0.set_title('Output membership activity')

for ax in (ax0, ax1, ax2):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()

# Result
aggregated = np.fmax(rule1, np.fmax(rule2, rule3))
def_prob = fuzz.defuzz(prob, aggregated, 'centroid')
res_prob = fuzz.interp_membership(prob, aggregated, def_prob)

# Visualization
fig3, ax0 = plt.subplots(figsize=(7, 3))
ax0.plot(prob, low_prob, 'b', linewidth=0.5, linestyle='--', )
ax0.plot(prob, aver_prob, 'r', linewidth=0.5, linestyle='--', )
ax0.plot(prob, high_prob, 'y', linewidth=0.5, linestyle='--', )
ax0.fill_between(prob, prob0, aggregated, facecolor='Orange', alpha=0.7)
ax0.plot([def_prob, def_prob], [0, res_prob], 'k', linewidth=1.5, alpha=0.9)
ax0.set_title('Aggregated membership and result (line)')

for ax in (ax0, ax1, ax2):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()