letters = [' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

num_classes = len(letters) + 1

word_cfg = {
	'batch_size': 64,
	'input_length': 30,
	'model_name': 'iam_words',
	'max_text_len': 16,
	'img_w': 128,
	'img_h': 64
}

line_cfg = {
	'batch_size': 16,
	'input_length': 98,
	'model_name': 'iam_line',
	'max_text_len': 74,
	'img_w': 800,
	'img_h': 64
}