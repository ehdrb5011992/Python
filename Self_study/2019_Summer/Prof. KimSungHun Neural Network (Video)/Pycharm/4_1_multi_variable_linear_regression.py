# Lab 4 Multi-variable linear regression# Has the distribution of manufacturer changed?
# count_top_7_planes<- dat2_2_2 %>%  head(7)
# count_the_other_sum_planes<- dat2_2_2[8:nrow(dat2_2_2),] %>% select(count) %>% sum()
# other1 <- tibble(manufacturer = "OTHER" , count=count_the_other_sum_planes)
# dat2_2_3_planes <- count_top_7_planes %>%  rbind(other1) %>% mutate(prop = round(count / sum(count) , 2))
#
#
# flights_planes_tailnum <- flights %>% inner_join(planes,by='tailnum')
# flights_manufacturer_count <- flights_planes_tailnum %>% select(manufacturer) %>% group_by(manufacturer) %>% summarize(count=n()) %>% arrange(desc(count))
# count_top_7_flights <- flights_manufacturer_count %>% head(7)
# count_the_other_sum_flights <- flights_manufacturer_count[8:nrow(flights_manufacturer_count),] %>% select(count) %>% sum()
# other2 <- tibble(manufacturer = "OTHER" , count=count_the_other_sum_flights)
# dat2_2_3_flights <- count_top_7_flights %>%  rbind(other2) %>% mutate(prop = round(count / sum(count) , 2))
#
#
# temp <- dat2_2_3_planes %>% left_join(dat2_2_3_flights , by = "manufacturer") %>%
#   select(manufacturer,prop.x , prop.y) %>% rename(Data_planes= prop.x ,Data_flights= prop.y )
# final_dat2_2_3 <- temp %>% gather('Data_planes','Data_flights',key="Data_from",value="prop")
#
# #plot
# final_dat2_2_3 %>% ggplot(aes(x=manufacturer,y=prop,group=Data_from)) +
#
#   #data
#   geom_bar(aes(fill=Data_from),stat='identity',position='dodge',width=0.7) +
#
#   #text
#   geom_text(aes(label=final_dat2_2_3$prop), position = position_dodge(width=0.7),vjust=-0.5) +
#
#   #axis
#   labs(x="",y="") +
#   scale_x_discrete(label = final_dat2_2_3$manufacturer) +
#
#   #title & caption
#   ggtitle("Changes in the Manufacturer Distribution" ,subtitle = "Flights Data & Planes Data : Top 7 , Others") +
#   labs(caption = "Made by Lee Dong Gyu" ) +
#
#   #theme
#   theme(plot.title = element_text(face = "bold", size =30) ) +
#   theme(plot.subtitle = element_text(face = "bold", size =15) ) +
#   theme(plot.caption = element_text(face="italic")) +
#   theme(axis.text.x = element_text( face = "bold",angle=45,hjust=1)) +
#   theme(axis.text.y = element_text( face = "bold")) +
#
#   geom_hline(yintercept=0 , color = '#545454',size=1)
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]

y_data = [152., 185., 180., 196., 142.]

# placeholders for a tensor that will be always fed.
x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)

Y = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.Variable(tf.random_normal([1]), name='weight2')
w3 = tf.Variable(tf.random_normal([1]), name='weight3')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b #bias 추가

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize. Need a very small learning rate for this data set
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
#0.01 말고 1e-5정도로 줌.
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                   feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)


###위처럼 하지말자. 아래처럼 행렬을 쓰자.###


# Lab 4 Multi-variable linear regression
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]
y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]
#데이터는 실수타입. 정수형 체크해보기.


# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 3])
#shape에 none을 주고 데이터를 무한정 받을수 있다는것에 주목하자.
#n개를 지닐것이다라는 것을 의미함. numpy에서는 이게 -1을 넣으면 됨.
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')

#tf 에서는 random_normal 난수임.
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
hypothesis = tf.matmul(X, W) + b
#matmul에 대해서 이해해보자.(찾아보자.)

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)