import torch
from skrl.resources.preprocessors.torch import RunningStandardScaler
from torch import Tensor
from typing import Optional, Tuple, Union
from skrl.memories.torch import RandomMemory


class ImageRunningStandardScaler(RunningStandardScaler):
    def __init__(
        self,
        size,
        device,
        epsilon: float = 1e-8,
        clip_threshold: float = 5,
    ) -> None:
        self.size = size
        size = self.size.spaces["state"]
        super().__init__(size, epsilon, clip_threshold, device)

    def _compute(self, x: Tensor, train: bool = False, inverse: bool = False) -> Tensor:

        state_num = self.size.spaces["state"].shape[0]
        state_x = x[:, -state_num:]
        if train:
            if state_x.dim() == 3:
                self._parallel_variance(
                    torch.mean(state_x, dim=(0, 1)), torch.var(state_x, dim=(0, 1)), state_x.shape[0] * state_x.shape[1]
                )
            else:
                self._parallel_variance(torch.mean(state_x, dim=0), torch.var(state_x, dim=0), state_x.shape[0])

        # scale back the data to the original representation
        if inverse:
            return (
                torch.sqrt(self.running_variance.float())
                * torch.clamp(state_x, min=-self.clip_threshold, max=self.clip_threshold)
                + self.running_mean.float()
            )
        # standardization by centering and scaling
        standard_state = torch.clamp(
            (state_x - self.running_mean.float()) / (torch.sqrt(self.running_variance.float()) + self.epsilon),
            min=-self.clip_threshold,
            max=self.clip_threshold,
        )
        res = torch.cat((x[:, :-state_num], standard_state), dim=1)
        return res


class CpuRandomMemory(RandomMemory):
    def __init__(
        self,
        memory_size: int,
        num_envs: int = 1,
        device: Optional[Union[str, torch.device]] = None,
        export: bool = False,
        export_format: str = "pt",
        export_directory: str = "",
        replacement=True,
    ):
        super().__init__(memory_size, num_envs, device, export, export_format, export_directory, replacement)

    def sample(self, names: Tuple[str], batch_size: int, mini_batches: int = 1, sequence_length: int = 1):
        """Sample a batch from memory randomly

        :param names: Tensors names from which to obtain the samples
        :type names: tuple or list of strings
        :param batch_size: Number of element to sample
        :type batch_size: int
        :param mini_batches: Number of mini-batches to sample (default: ``1``)
        :type mini_batches: int, optional
        :param sequence_length: Length of each sequence (default: ``1``)
        :type sequence_length: int, optional

        :return: Sampled data from tensors sorted according to their position in the list of names.
                 The sampled tensors will have the following shape: (batch size, data size)
        :rtype: list of torch.Tensor list
        """
        # compute valid memory sizes
        size = len(self)
        if sequence_length > 1:
            sequence_indexes = torch.arange(0, self.num_envs * sequence_length, self.num_envs)
            size -= sequence_indexes[-1].item()

        # generate random indexes
        if self._replacement:
            indexes = torch.randint(0, size, (batch_size,))
        else:
            # details about the random sampling performance can be found here:
            # https://discuss.pytorch.org/t/torch-equivalent-of-numpy-random-choice/16146/19
            indexes = torch.randperm(size, dtype=torch.long)[:batch_size]

        # generate sequence indexes
        if sequence_length > 1:
            indexes = (sequence_indexes.repeat(indexes.shape[0], 1) + indexes.view(-1, 1)).view(-1)

        self.sampling_indexes = indexes
        res = self.sample_by_index(names=names, indexes=indexes, mini_batches=mini_batches)
        res = [[i.to("cuda") for i in _resj] for _resj in res]
        return res
