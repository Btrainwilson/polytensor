import numpy as np
import torch


        if len(self.coefficients) == 0:
            raise ValueError("Coefficients cannot be empty.")

        for term, value in self.coefficients.items():
            if type(value) not in [
                int,
                float,
                complex,
                torch.Tensor,
                np.ndarray,
            ]:
                raise TypeError(
                    "Coefficients must be a number, numpy array, or a tensor."
                )
            for t in zip(
                [int, float, complex], [torch.int, torch.float, torch.complex]
            ):
                if type(value) == t[0]:
                    if self.dtype != t[1]:
                        logging.warning(
                            f"Coefficient {term} is type {type(value)} and will be converted to type {self.dtype}."
                        )

            if len(term) > 1 and (np.diff(term) < 0).all():
                raise ValueError(
                    f"Coefficients {np.diff(term)} must be in non-decreasing order."
                )

        # Make sure all coefficients are unique
        if len(self.coefficients.keys()) != len(set(self.coefficients.keys())):
            raise ValueError("Coefficients must be unique.")

        self.coeff_vector = torch.nn.ParameterList()
        self.coeff_map = {}
        for i, term in enumerate(self.coefficients.keys()):
            self.coeff_map[term] = i
            if type(value) not in [torch.Tensor]:
                self.coeff_vector.append(
                    torch.nn.Parameter(
                        torch.tensor(
                            self.coefficients[term],
                            dtype=self.dtype,
                            device=self.device,
                        )
                    )
                )
            else:
                self.coeff_vector.append(
                    torch.nn.Parameter(
                        self.coefficients[term].to(self.dtype).to(self.device)
                    )
                )

        return True

