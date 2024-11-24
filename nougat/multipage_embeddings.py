from torch import randn
from torch import nn

class MultipageEmbeddings(nn.Linear):

    def __init__(self, max_page_count=40, embedding_size=1024, tokens_per_page=588):
        super().__init__()
        self.page_embeddings = nn.Parameter(randn(1, max_page_count, embedding_size))
        self.intra_page_embeddings = nn.Parameter(randn(1, tokens_per_page, embedding_size))

    def forward(self, x):

        page_count = x.shape[1]

        # this is because we want every token of each page to have the token embeddings for that page
        x+=self.page_embeddings[:, :page_count].repeat_interleave(self.intra_page_embeddings.shape[1], dim=1)

        # this is because we want all three pages to have the same token embeddings intra-page
        # so we repeat the token embeddings for each page (second dimension of the page embedding
        # tensor is the page count, as we can see in the init function)
        x+=self.intra_page_embeddings.repeat(1, page_count, 1)


        return x
