from utils import detect_language

def translate_prompt(text, target_language):
    trans = {
        'Question: ': {
            'EN': 'Question: ',
            'ZH': '问题：',
            'AR': 'سؤال:',
            'RU': 'вопрос:'
        },
        'Hint: ': {
            'EN': 'Hint: ',
            'ZH': '提示：',
            'AR': 'تَلمِيح:',
            'RU': 'намекать:'
        },
        'Answer: ':{
            'EN': 'Answer: ',
            'ZH': '答案: ',
        },
        'Output: ':{
            'EN': 'Your output: ',
            'ZH': '你的输出: ',
        }
    }
    return trans[text][target_language]


TYPE2LANGUAGE2PROMPT = {
    'multiple_choice': {
        'EN': 'Please select the correct answer from the options above.\n',
        'ZH': '请直接选择正确选项的字母。\n',
        'AR': 'الرجاء اختيار الإجابة الصحيحة من الخيارات أعلاه.\n',
        'RU': 'Пожалуйста, выберите букву правильного варианта напрямую.\n'
    },
    'open': {
        'EN': 'Please answer the question directly.\n',
        'ZH': '请直接回答问题。\n',
        'AR': 'يرجى الإجابة على السؤال مباشرة.\n',
        'RU': 'Пожалуйста, ответьте на вопрос прямо.\n'
    },
    'yes_or_no': {
        'EN': 'Please answer Yes or No.\n',
        'ZH': '请回答是或否。\n',
        'AR': 'من فضلك أجب بنعم أو لا.\n',
        'RU': 'Пожалуйста, ответьте Да или Нет.\n',
    },
    'cot': {
        'EN': 'Let\'s think step by step. ',
        'ZH': '让我们一步一步来思考。',
        'AR': 'دعنا نفكر خطوة بخطوة.',
        'RU': 'Давайте думать шаг за шагом.'
    }
}

FILTER_TYPE2LANGUAGE2PROMPT = {
    'multiple_choice': {
        'EN': '''
            Please help me match an answer to the multiple choices of the question.
            The option is ABCDEF.
            You get a question and an answer,
            You need to figure out options from ABCDEF that is most similar to the answer.
            If the meaning of all options is significantly different from the answer, we output Unknown.
            You should output one or more options from the following :ABCDEF.
            Example 1:
            Question: Which of the following numbers are positive? \nA.0\nB.-1\nC.5\nD.102\nE.-56\nF.33
            Answer: The answers are 5,102 and 33. \nYour output:CDF
            Example 2:
            Question: Which of these countries is landlocked? \nA.Mongolia \nB.United States \nC.China \nD.Japan
            Answer: A.Mongolia is A landlocked country. \nYour output:A
            Here are the target questions and answers.\n
        ''',

        'ZH': f'''
            请帮我把一个答案与问题的多个选项相匹配。
            选项有<possibilites>。
            你会得到一个问题和一个答案，
            你需要找出哪几个选项<possibilites>与答案最相似。
            如果所有选项的含义都与答案显著不同，则输出Unknown。
            你应该从以下几个个选项中输出一个或多个选项:<possibilites>。
            示例1:
            问题:下面那几个数字是正数？\nA.0\nB.-1\nC.5\nD.102\nE.-56\nF.33
            答案:答案是5，102和33。\n你的输出:CDF
            示例2:
            问题:下面哪个国家是内陆国家?\nA.蒙古\nB.美国\nC.中国\nD.日本
            答案：A.蒙古国是内陆国家。\n你的输出:A
            下面是目标的问题和答案。\n
        ''',
    },
    'open': {
        'EN': '''
        Please help me extract the answers to the given questions from the answers given.
        You get a question and an answer,
        If the answer and question are not relevant, the output is Unknown.
        Example 1:
        Question: What color is the sky?
        Answer: The sky is blue most of the time.\nYour output: The sky is blue
        Example 2:
        Question: Who invented the electric light?
        Answer: Edison is often credited with inventing the light bulb. In fact, he only improved it. The actual inventor of the light bulb was Henry Goebbels.\nYour output: Henry Goebbels
        Here are the target questions and answers. \n
        ''',
        'ZH': '''
            请帮我从给出的答案中提取出给出问题的回答。
            你会得到一个问题和一个答案，
            如果答案和问题不相关，则输出Unknown。
            示例1:
            问题:天空是什么颜色的？
            答案:天空是在大多数时候是蓝色的。\n你的输出:天空是蓝色的
            示例2:
            问题:是谁发明了电灯？
            答案：人们通常认为是爱迪生发明了电灯泡，实际上不然，他只是改进了电灯泡。电灯泡的实际发明人是亨利·戈培尔。\n你的输出:亨利·戈培尔
            下面是目标的问题和答案。\n
        '''
    },
    'yes_or_no': {
        'ZH': '''
            请帮我把一个答案与问题的两个选项相匹配。
            选项只有“是/否”。
            你会得到一个问题和一个答案，
            你需要找出哪个选项(是/否)与答案最相似。
            如果所有选项的含义都与答案显著不同，则输出Unknown。
            你应该从以下3个选项中输出一个单词:Yes, No, Unknown。
            示例1:
            问题:图像中的单词是“Hello”吗?
            答案:这个图像中的单词是“Hello”。\n你的输出:Yes
            示例2:
            问题:图像中的单词是“Hello”吗?
            答案:这个图像中的单词不是“Hello”。\n你的输出:No
            下面是目标的问题和答案。\n
        ''',

        'EN': '''
            Please help me to match an answer with two options of a question.
            The options are only Yes / No.
            You are provided with a question and an answer,
            and you need to find which option (Yes / No) is most similar to the answer.
            If the meaning of all options are significantly different from the answer, output Unknown.
            Your should output a single word among the following 3 choices: Yes, No, Unknown.
            Example 1:
            Question: Is the word in this image 'Hello'?
            Answer: The word in this image is 'Hello'.\nYour output: Yes
            Example 2:
            Question: Is the word in this image 'Hello'?
            Answer: The word in this image is not 'Hello'.\nYour output: No\n
            Now here are the target's Question and Answer:\n
        '''
    },
}
