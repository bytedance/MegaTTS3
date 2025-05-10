# -*- coding: utf-8 -*-
# Copyright 2025 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

def chunk_text_chinese(text, limit=60):
    # 中文字符匹配
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    # 标点符号匹配
    punctuation = "，。！？；：,\.!?;"
    
    result = []  # 存储断句结果
    current_chunk = []  # 当前片段
    chinese_count = 0  # 中文字符计数

    i = 0
    while i < len(text):
        char = text[i]
        current_chunk.append(char)
        if chinese_pattern.match(char):
            chinese_count += 1
        
        if chinese_count >= limit:  # 达到限制字符数
            # 从当前位置往前找最近的标点符号
            for j in range(len(current_chunk) - 1, -1, -1):
                if current_chunk[j] in punctuation:
                    result.append(''.join(current_chunk[:j + 1]))
                    current_chunk = current_chunk[j + 1:]
                    chinese_count = sum(1 for c in current_chunk if chinese_pattern.match(c))
                    break
            else:
                # 如果前面没有标点符号，则继续找后面的标点符号
                for k in range(i + 1, len(text)):
                    if text[k] in punctuation:
                        result.append(''.join(current_chunk)+text[i+1:k+1])
                        current_chunk = []
                        chinese_count = 0
                        i = k
                        break
        i+=1

    # 添加最后剩余的部分
    if current_chunk:
        result.append(''.join(current_chunk))

    return result

def chunk_text_english(text, max_chars=130):
    """
    Splits the input text into chunks, each with a maximum number of characters.

    Args:
        text (str): The text to be split.
        max_chars (int): The maximum number of characters per chunk.

    Returns:
        List[str]: A list of text chunks.
    """
    chunks = []
    current_chunk = ""
    # Split the text into sentences based on punctuation followed by whitespace
    sentences = re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text)

    for sentence in sentences:
        if len(current_chunk.encode("utf-8")) + len(sentence.encode("utf-8")) <= max_chars:
            current_chunk += sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def chunk_text_chinesev2(text, limit=60, look_ahead_limit=30):
    """
    将中文文本分成多个块，优先确保每个块以句号、感叹号或问号结尾，
    其次考虑逗号等其他标点符号，避免在无标点处断句
    
    参数:
        text: 要分块的文本
        limit: 每个块的中文字符数限制
        look_ahead_limit: 向后查找的最大字符数限制
    
    返回:
        分块后的文本列表
    """
    # 中文字符匹配
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    
    # 分级定义标点符号（优先级从高到低）
    primary_end_marks = "。.！!？?"  # 首选：句号、感叹号、问号
    secondary_end_marks = "，,；;："  # 次选：逗号、分号、冒号
    tertiary_end_marks = "、…—-~～"  # 再次：顿号、省略号、破折号等
    
    result = []  # 存储断句结果
    current_chunk = []  # 当前片段
    chinese_count = 0  # 中文字符计数

    i = 0
    while i < len(text):
        char = text[i]
        current_chunk.append(char)

        if chinese_pattern.match(char):
            chinese_count += 1
        
        if chinese_count >= limit:  # 达到字符数限制，需要寻找断句点
            found_end = False
            
            # 依次尝试不同优先级的断句策略
            
            # 1. 向后查找首选标点
            for k in range(1, min(look_ahead_limit, len(text) - i)):
                next_char = text[i + k]
                if next_char in primary_end_marks:
                    result.append(''.join(current_chunk) + text[i+1:i+k+1])
                    current_chunk = []
                    chinese_count = 0
                    i = i + k
                    found_end = True
                    break
            
            if not found_end:
                # 2. 向前查找首选标点
                for j in range(len(current_chunk) - 1, -1, -1):
                    if current_chunk[j] in primary_end_marks:
                        result.append(''.join(current_chunk[:j + 1]))
                        current_chunk = current_chunk[j + 1:]
                        chinese_count = sum(1 for c in current_chunk if chinese_pattern.match(c))
                        found_end = True
                        break
            
            if not found_end:
                # 3. 向后查找次选标点
                for k in range(1, min(look_ahead_limit, len(text) - i)):
                    next_char = text[i + k]
                    if next_char in secondary_end_marks:
                        result.append(''.join(current_chunk) + text[i+1:i+k+1])
                        current_chunk = []
                        chinese_count = 0
                        i = i + k
                        found_end = True
                        break
            
            if not found_end:
                # 4. 向前查找次选标点
                for j in range(len(current_chunk) - 1, -1, -1):
                    if current_chunk[j] in secondary_end_marks:
                        result.append(''.join(current_chunk[:j + 1]))
                        current_chunk = current_chunk[j + 1:]
                        chinese_count = sum(1 for c in current_chunk if chinese_pattern.match(c))
                        found_end = True
                        break
            
            if not found_end:
                # 5. 向后查找三级标点
                for k in range(1, min(look_ahead_limit, len(text) - i)):
                    next_char = text[i + k]
                    if next_char in tertiary_end_marks:
                        result.append(''.join(current_chunk) + text[i+1:i+k+1])
                        current_chunk = []
                        chinese_count = 0
                        i = i + k
                        found_end = True
                        break
            
            if not found_end:
                # 6. 向前查找三级标点
                for j in range(len(current_chunk) - 1, -1, -1):
                    if current_chunk[j] in tertiary_end_marks:
                        result.append(''.join(current_chunk[:j + 1]))
                        current_chunk = current_chunk[j + 1:]
                        chinese_count = sum(1 for c in current_chunk if chinese_pattern.match(c))
                        found_end = True
                        break
            
            if not found_end:
                # 万不得已，在此处断句（这种情况很少见，因为汉语文本中通常会有标点）
                result.append(''.join(current_chunk))
                current_chunk = []
                chinese_count = 0
        
        i += 1

    # 添加最后剩余的部分
    if current_chunk:
        result.append(''.join(current_chunk))

    # 英文标点替换为中文标点
    punctuation_map = {
        '.': '。',
        ',': '，',
        '!': '！',
        '?': '？',
        ';': '；',
        ':': '：'
    }
    
    for i in range(len(result)):
        for eng_punc, cn_punc in punctuation_map.items():
            result[i] = result[i].replace(eng_punc, cn_punc)

    return result

if __name__ == '__main__':
    print(chunk_text_chinese("哇塞！家人们，你们太好运了。我居然发现了一个宝藏零食大礼包，简直适合所有人的口味！有香辣的，让你舌尖跳舞；有盐焗的，咸香可口；还有五香的，香气四溢。就连怀孕的姐妹都吃得津津有味！整整三十包啊！什么手撕蟹柳、辣子鸡、嫩豆干、手撕素肉、鹌鹑蛋、小肉枣肠、猪肉腐、魔芋、魔芋丝等等，应有尽有。香辣土豆爽辣过瘾，各种素肉嚼劲十足，鹌鹑蛋营养美味，真的太多太多啦，...家人们，现在价格太划算了，赶紧下单。"))
    print(chunk_text_english("Washington CNN When President Donald Trump declared in the House Chamber this week that executives at the nation’s top automakers were “so excited” about their prospects amid his new tariff regime, it did not entirely reflect the conversation he’d held with them earlier that day."))
    text = "欢迎收听《TED Talks Daily》，在这里，我们每天为您带来新思想，激发您的好奇心。我是您的主持人，Elise Hugh。当我们去看医生时，医生会评估我们的身体健康状况，检查我们的生命体征，可能还会关注我们的胆固醇水平，确保我们整体处于健康状态。医生可能还会通过一系列问题来检查我们的心理健康。然而，人际交往专家Casley Killam指出，我们在理解健康时忽略了一个关键指标，那就是我们的社会健康。在2024年的演讲中，她解释了为什么人际关系如此重要，以及忽视它可能带来的代价。几年前，我认识的一位女士，我们暂且称她为Maya，在短时间内经历了许多重大变化。她结婚了，和丈夫因工作搬到了一个陌生的城市，在那里她谁也不认识。她开始了一份在家办公的新工作，同时还要应对父亲新确诊的痴呆症。为了应对这些变化带来的压力，Maya加倍关注自己的身心健康。她几乎每天都锻炼，吃健康的食物，每周去看一次心理医生。这些措施确实有帮助，她的身体变得更加强壮，心理也更具韧性，但效果有限。她仍然感到困扰，经常在半夜失眠，白天感到注意力不集中，缺乏动力。Maya做了医生通常建议我们做的所有事情来保持身心健康，但似乎还缺少些什么。如果我告诉你，Maya所缺少的东西，也是全球数十亿人所缺少的，甚至可能也是你所缺少的呢？如果我告诉你，缺乏它会削弱我们为保持健康所做的其他努力，甚至可能缩短你的寿命呢？我研究这个问题已经超过十年，我发现，我们传统上对健康的理解是不完整的。通过将健康主要视为身体和心理的健康，我们忽略了我认为是我们这个时代最大的挑战和机遇——社会健康。身体健康关乎我们的身体，心理健康关乎我们的思想，而社会健康则关乎我们的人际关系。如果你以前没有听说过这个词，那是因为它还没有进入主流词汇，但它同样重要。Maya在她的新家还没有归属感。她不再亲自见到她的家人、朋友或同事，她经常一连几周只和丈夫共度时光。她的故事告诉我们，如果我们只照顾身体和心理，而不关注人际关系，我们就无法完全健康，无法真正茁壮成长。与Maya类似，全球有数亿人连续几周不与任何朋友或家人交谈。全球范围内，有四分之一的人感到孤独。20%的成年人觉得他们没有任何人可以求助。想想看，你遇到的每五个人中，可能有一个人觉得自己孤立无援。这不仅令人心碎，也是一场公共卫生危机。"
    for res in chunk_text_chinesev2(text):
        print(res)
