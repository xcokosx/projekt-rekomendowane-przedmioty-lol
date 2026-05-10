from app import siec
model = siec.ItemRanker.load('./app/model_retrained.json')
# sample
sample = {'champion':'Aphelios','role':'BOT','ally_team':['Lulu','Ahri','Cho\'Gath','Xin Zhao'],'enemy_team':['Azir','Teemo','Karma','Twitch','Volibear'],'items':[]}
context = siec.build_context(sample)
print('MAX_ITEMS', siec.MAX_ITEMS, 'len(item_to_id)', len(siec.item_to_id))
# probe a range of items
candidates = [3003,667666,1011,1031,1043,1053]
for it in candidates:
    idx = siec.encode_item(it)
    x = context + siec.pad_build([]) + [siec.encode_item(it)/ max(1, siec.MAX_ITEMS)]
    out = model.forward(x)
    print(it, 'idx=', idx, 'out_raw=', getattr(model,'out_raw',None), 'out=', out)
# also check a few arbitrary candidates from all_items
print('\nSome all_items samples:')
for it in siec.all_items[:10]:
    idx = siec.encode_item(it)
    x = context + siec.pad_build([]) + [siec.encode_item(it)/ max(1, siec.MAX_ITEMS)]
    out = model.forward(x)
    print(it, 'idx=', idx, 'out=', out)
