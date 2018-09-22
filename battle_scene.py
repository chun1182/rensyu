import random

my_hit_point = 15
slime_hit_point = 10

index = 0
while my_hit_point > 0 and slime_hit_point > 0:
  attack = random.randint(1,7)
  if index % 2 == 1:
      print('スライムに' + str(attack) + 'のダメージ')
      slime_hit_point -= attack
  else:
      print('ゆうしゃに' + str(attack) + 'のダメージ')
      my_hit_point -= attack
  index += 1
if my_hit_point > 0:
    print('スライムをやっつけた')
else:
    print('ゆうしゃは死んでしまった')
