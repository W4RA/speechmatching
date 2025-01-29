"""Simple example of a Telegram bot."""

import json
import logging
import os

import telebot

from speechmatching import Group, Recording

# create the Telegram bot with the token available in the environment.
bot = telebot.TeleBot(os.environ['TELEGRAM_BOT_TOKEN'], parse_mode=None)
telebot.logger.setLevel(logging.DEBUG)

# states:
#   0: written command mode
#   1: expecting recording to register
#   2: expecting recording to match
state = 0
groups = []

# the data directory to store the recordings and their related data in
DATA_DIR = os.environ.get('TELEGRAM_DATA_DIR') or 'data'


def clean_group(message):
    """Remove the last added incomplete group of recordings.

    Args:
        message: The message to reply to
    """
    global groups
    if len(groups) > 0 and len(groups[-1]) == 0:
        bot.reply_to(message, 'Last group has no recordings, removing group.')
        groups = groups[:-1]


@bot.message_handler(commands=['new'])
def new(message):
    """Handle the ``\\new``` command.

    This function removes all stored group, it effectively empties the list of
    groups, and sets the state to be ready for receiving words.

    Args:
        message: The message to reply to.
    """
    global groups
    global state
    groups = []
    bot.reply_to(message, 'Ready for new words.')
    state = 0


@bot.message_handler(commands=['interpret'])
def interpret(message):
    """Handle the ``\\interpret``` command.

    Changes from the data recording state to the interpretation state, in which
    recordings of words can be sent and matched with the previously registered
    groups.

    To move to this state, the bot requires are least two words to be
    registered, else there is nothing to make a decision between.

    Args:
        message: The message to reply to.
    """
    global groups
    global state
    if state == 2:
        bot.reply_to(message, 'Already in interpretation mode.')
        return None
    if len(groups) < 2:
        bot.reply_to(message, 'Need at least two words. Only {} given.'.format(len(groups)))
        return None
    if state == 1:
        clean_group(message)
    state = 2
    bot.reply_to(message, 'Switched to interpretation mode with\n\t'+'\n\t'.join('{}: {} recordings'.format(group.identifier, len(group)) for group in groups)+'\nSpeak a registered word.')


@bot.message_handler(content_types=['voice'])
def audio(message):
    """Handle the a voice message.

    A voice message is handled depending on the state the bot is in. If the bot
    is in state ``0``, the bot is not ready to receive a voice message and it
    replies as such. If the bot is in state ``1``, the bot will register the
    recording with the previously typed word. If the bot is in state ``2``, the
    voice message will be matched against the previously given groups and the
    best match will be replied with, or no decision can be made.

    Args:
        message: The message to reply to.
    """
    global groups
    global state
    if state == 0:
        bot.reply_to(message, 'Not ready to receive voice message.')
        return None
    file_info = bot.get_file(message.voice.file_id)
    data_path = os.path.join(DATA_DIR, str(message.chat.id), str(message.message_id))
    if not os.path.isdir(data_path):
        os.makedirs(data_path)
    with open(os.path.join(data_path, 'message.json'), 'w') as f:
        json.dump(message.json, f)
    filepath = os.path.join(data_path, file_info.file_path.split('/')[-1])
    with open(filepath, 'wb') as f:
        f.write(bot.download_file(file_info.file_path))
    recording = Recording(filepath)
    print(recording.transcript.text)
    # match with the previously registered groups
    if state == 2:
        match = recording.match(groups, return_indecision=True)
        if match is None:
            bot.reply_to(message, 'Could not choose a best match.')
            return None
        bot.reply_to(message, 'Matching \'{}\'.'.format(match.identifier))
    # register the voice message with the previous given word
    elif state == 1:
        groups[-1].add(recording)
        bot.reply_to(message, 'Registered speech with \'{}\'.\nType a new word, or speak to add another recording.'.format(groups[-1].identifier))


@bot.message_handler(func=lambda message: True)
def echo_all(message):
    """Handle a text message.

    When a simple text message is given to the bot, meaning a message that does
    not start with ``\\`` and so is not a command, the bot will handle this
    according to the state it is in. If the bot is in state ``0``, the bot will
    register the text as representation of the next voice message. If the bot
    is in state ``1``, it is actually waiting for one or more voice message,
    but will instead remove the last given word if no voice messages were given
    and move on to register the given text as new word to wait for voice
    messages for. If the bot is in state ``2``, it is in interpretation mode
    and will return an error.

    Args:
        message: The message to reply to.
    """
    global groups
    global state
    if state == 1:
        clean_group(message)
        state = 0
    if state != 0:
        bot.reply_to(message, 'Not ready to receive word.')
        return None
    groups.append(Group(identifier=message.text))
    bot.reply_to(message, 'Speak to add a recording.')
    state = 1

bot.infinity_polling()

