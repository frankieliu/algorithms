create table jobs_250713_filtered as
select * 
from jobs_250713
where action <> 0
-- order by score desc, status asc, `title-short`
order by comment, pay_max
;