Telegram Bot
############

This is a simple example of a Telegram bot. The bot can receive text and voice
message to create groups of recordings, and can receive voice messages to match
against these groups to find the best matching group.

The message expected by the bot depends on the state it is in. There are three
states:

 - Text state: the bot expects a text as representation of a word that is about
   to be registered. After receiving this word, the bot will move to the next
   state in this list.

 - Recording registration state: the bot expects a voice message to register with
   the previously given textual representation of it. One or more recording
   may be given after each other. If instead of a recording, a text message is
   received, the bot will abandon the previous word if not recordings were
   registered to it, or keep it, and more to the "text state".

 - Interpretation state: the bot expects a voice message to match against
   the previously registered recordings.

The commands to move between the states are as follows:

 - ``\new``: remove all recordings and switch to the text state.

 - ``\interpret``: move from the test or recording registration state to the
   interpretation state.

Setup
*****

For the Telegram bot, one needs to get a token from Telegram for the `guide`_
can be followed. When this token is available, change the name of file
``env.list.CHANGEME`` to ``env.list``, and replace dummy token
``123456789:abcdefghijklmnopqrstuvwxyzABCDEFGH`` by the token from Telegram.

When the token is in place, the bot can be run manually or using Docker.

.. _guide: https://core.telegram.org/bots/features#botfather

Manual
======

Please see the manual setup in the README for the Basic example. The setup for
the Telegram bot requires one extra step, setting the environment variable from
``env.list``. Right after loading the Python virtual environment using
``source env/bin/activate``, the Telegram bot token can be loaded into the
environment with::

    set -a
    source env.list
    set +a

After which the instructions from the setup for the Basic example are the same.

Docker
======

As for the Basic example, the Telegram bot can also be run using Docker.

Build the image for the Telegram bot under name ``speechmatching-bot``::

    docker build . -t speechmatching-bot

After this completes successfully, run the image using::

    docker run --rm --env-file env.list -it speechmatching-bot

where the previously filled in ``env.list`` file is used.

