drop table jobs_2025_06_02_26500_status;
create table jobs_2025_06_02_26500_status as
select s.number, s.`title-short`, s.team, s.title, s.location, s.posted, s.summary, s.description, s.minimum, s.pref, s.pay_min, s.pay_max, t.status, t.score, t.comment
from `jobs.2025.06.02` as s
inner join jobs_2025_06_02_265000 as t
on s.number = t.number
where s.pay_max > 265000
order by s.`title-short`, s.team, s.posted desc, s.pay_max desc
;
