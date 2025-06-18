-- drop table jobs_2025_06_02_26500_score_6;
create table jobs_250617_sorted as
select * 
from jobs_250617
-- where score > 5
order by score desc, status asc, `title-short`
;