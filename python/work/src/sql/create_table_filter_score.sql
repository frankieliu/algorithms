drop table jobs_2025_06_02_26500_score_6;
create table jobs_2025_06_02_26500_score_6 as
select * 
from jobs_2025_06_02_26500_status
where score > 5
order by score desc, status asc, `title-short`
;