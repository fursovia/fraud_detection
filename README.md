# Demonstrative data

|   | id         | amount  |  age        | sex  | ins_type | speciality | treatments                                     | types                     | target |
|---|----------- |---------| ------------|------|----------|------------|------------------------------------------------|---------------------------|--------|
| 0 | ID_221227  |  2.68   |  44.609818  |  1   |    1     |     1      | ['A_852']                                      | ['AA_13']                 |   0    |
| 1 | ID_188617	 |  67.70  |  58.471695  |  1   |    1     |     1      | ['A_645', 'A_2030', 'A_1978']                  | ['AA_2', 'AA_7', 'AA_2']	 |   0    |
| 2 |	ID_184107  |  21.44	 |  48.003800  |  1   |    1     |     1      | ['A_1656', 'A_1']	                             | ['AA_2', 'AA_2']	         |   0    |
| 3 |	ID_327112  |  26.80  |  60.127670  |  0   |    1     |     1      | ['A_1', 'A_348', 'A_1656']	                   | ['AA_2', 'AA_12', 'AA_2'] |   0    |

`demanstrative_data.csv` file - example of an used data

Meaning of the columns:
* `id` - patient id
* `amount` - the total billable amount
* `age` - patient age
* `sex` - patient sex
* `ins_type` - insurance type
* `speciality` - doctor specialty
* `treatments` - a sequence of treatments encoded with anonymized IDs
* `types` - corresponding types of treatments
* `target` - label ("fraudulent" coded as 1 and "non-fraudulent" coded as 0)
