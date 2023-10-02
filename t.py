import torch


lista = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

t = torch.tensor(lista)

truths = [
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1],
]
truths = torch.tensor(truths, dtype='bool')
print(t[truths])
exit(-1)


t = torch.tensor([1, 2, 3])[[1, 1]]
t = torch.full((4,1), 1)

lista = range(27)
T = torch.tensor((lista))
T = T.reshape((3,3,3))


T = [
        [
            [10, 2, 3, 1],
            [4, 5, 6, 2],
            [7, 8, 9, 1]
        ],
        [
            [10, 11, 12, 8],
            [13, 18, 15, 7],
            [16, 17, 19, 2],
        ]
    ]
T = torch.tensor(T)

ind = (1, 2)
T0 = T[0]

print(T)



#just_maxes[torch.arange(indices[0, :], indices[:, ], torch.arange(just_maxes.shape[2])] = 1
# 0 0 0  0 2 1  0 2 2  0 1 0 

indices = [
    [0, 0, 0], 
    [0, 2, 1], 
    [0, 2, 2],
    [0, 1, 0],
]

manual_indices = torch.tensor([
    [0, 0, 0, 0], 
    [0, 2, 2, 1], 
    [0, 1, 2, 3]
])

manual_indices2 = torch.tensor([
    [0, 0, 0], 
    [0, 2, 1], 
    [0, 2, 2],
    [0, 1, 0],
])

manual_indices = list(manual_indices)

stagod = T[manual_indices]
print(f'stagod =\n  {stagod}')

just_maxes = torch.zeros_like(T) 
indices = torch.argmax(T, dim=1)

image_indices = torch.arange(just_maxes.shape[0]).repeat_interleave(just_maxes.shape[-1])
class_indices = torch.arange(indices.shape[-1]).repeat(just_maxes.shape[0])
print(f'\n\n img indices = {image_indices}')
print(f'\n\n img indices = {indices.flatten()}')
print(f'\n\n img indices = {class_indices}')
total_indices = torch.stack([image_indices, indices.flatten(), class_indices])
print(f'\n\n total_indices = {total_indices}')
#just_maxes[list(total_indices)] = 1
just_maxes[image_indices, indices.flatten(), class_indices] = 1
print(just_maxes)
exit(-1)
just_maxes[torch.arange(just_maxes.shape[0]), torch.arange(just_maxes.shape[1]), indices] = 1
just_maxes[torch.arange(just_maxes.shape[0]), torch.arange(just_maxes.shape[1]), indices] = 1

print(just_maxes)