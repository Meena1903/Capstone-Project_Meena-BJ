-- Q1: WHERE filters rows first, HAVING filters customer groups after
SELECT customer_id, ROUND(SUM(order_amount), 2) AS total_over_500
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_over_500 DESC;

-- Q2: revenue by city and product category (3-table join)
SELECT c.city, p.category, ROUND(SUM(o.order_amount), 2) AS revenue
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products  p ON o.product_id  = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY revenue DESC;

-- Q3: products priced above the catalogue average (scalar subquery)
SELECT product_id, product_name, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC;

-- Q4: customers with no completed order (condition lives in the ON clause)
SELECT c.customer_id, c.city
FROM customers c
LEFT JOIN orders o
       ON o.customer_id = c.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;

-- Q5: customers above the average customer spend (two CTEs)
WITH cust_total AS (
    SELECT customer_id, SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
avg_total AS (
    SELECT AVG(total_spend) AS avg_spend FROM cust_total
)
SELECT t.customer_id, ROUND(t.total_spend, 2) AS total_spend, ROUND(a.avg_spend, 2) AS avg_spend
FROM cust_total t, avg_total a
WHERE t.total_spend > a.avg_spend
ORDER BY t.total_spend DESC;

-- Q6: ranking window functions per city
WITH spend AS (
    SELECT c.customer_id, c.city, ROUND(SUM(o.order_amount), 2) AS total_spend
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.customer_id, c.city
)
SELECT customer_id, city, total_spend,
       RANK()       OVER (PARTITION BY city ORDER BY total_spend DESC) AS rnk,
       DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rnk,
       ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS row_num
FROM spend
ORDER BY city, row_num;

-- Q7: daily completed revenue with running total
WITH daily AS (
    SELECT SUBSTR(order_ts, 1, 10) AS day, ROUND(SUM(order_amount), 2) AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT day, revenue, ROUND(SUM(revenue) OVER (ORDER BY day), 2) AS cumulative_revenue
FROM daily
ORDER BY day;
